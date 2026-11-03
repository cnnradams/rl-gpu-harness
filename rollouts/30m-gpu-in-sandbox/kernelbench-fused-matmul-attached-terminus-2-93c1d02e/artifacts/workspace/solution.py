import torch
import torch.nn as nn
import triton
import triton.language as tl


def _alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device='cuda', dtype=torch.int8)


triton.set_allocator(_alloc_fn)


@triton.jit
def _split_kernel(X, Hi, Lo, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(X + offs, mask=mask)
    hi = x.to(tl.float16)
    lo = (x - hi.to(tl.float32)).to(tl.float16)
    tl.store(Hi + offs, hi, mask=mask)
    tl.store(Lo + offs, lo, mask=mask)


@triton.jit
def _gemm_fp16x3_tma(Ah, Al, Wh, Wl, Bias, C, M, N, K,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                     GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BM)
    grid_n = tl.cdiv(N, BN)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    off_m = pid_m * BM
    off_n = pid_n * BN
    a_desc = tl.make_tensor_descriptor(Ah, shape=[M, K], strides=[K, 1], block_shape=[BM, BK])
    al_desc = tl.make_tensor_descriptor(Al, shape=[M, K], strides=[K, 1], block_shape=[BM, BK])
    wh_desc = tl.make_tensor_descriptor(Wh, shape=[K, N], strides=[N, 1], block_shape=[BK, BN])
    wl_desc = tl.make_tensor_descriptor(Wl, shape=[K, N], strides=[N, 1], block_shape=[BK, BN])
    c_desc = tl.make_tensor_descriptor(C, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a_hi = a_desc.load([off_m, k])
        a_lo = al_desc.load([off_m, k])
        w_hi = wh_desc.load([k, off_n])
        w_lo = wl_desc.load([k, off_n])
        acc = tl.dot(a_hi, w_hi, acc)
        acc = tl.dot(a_hi, w_lo, acc)
        acc = tl.dot(a_lo, w_hi, acc)
    bias = tl.load(Bias + off_n + tl.arange(0, BN))
    acc = acc + bias[None, :]
    c_desc.store([off_m, off_n], acc)


_BM, _BN, _BK, _GM = 128, 128, 64, 8


class ModelNew(nn.Module):
    """
    Optimized reference model.

    Math: out = (x @ W^T + b) * (1 + s) = x @ (W*(1+s))^T + b*(1+s)
    The (1+s) factor is folded into the weights once.

    The FP32 GEMM is evaluated with an fp16 error-compensation scheme:
    each FP32 operand is split as v = v_hi + v_lo (both fp16), and
    x @ w ~= x_hi@w_hi + x_hi@w_lo + x_lo@w_hi accumulated in fp32.
    This gives ~fp32-level accuracy (error ~2^-22 relative) at fp16
    tensor-core speed on H100.
    """

    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self._ready = False

    def _prepare(self):
        s = 1.0 + float(self.scaling_factor)
        with torch.no_grad():
            self.matmul.weight.mul_(s)
            self.matmul.bias.mul_(s)
            wt = self.matmul.weight.t().contiguous()
            self._wt = wt
            self._w_hi_t, self._w_lo_t = self._split(wt)
            self._bias = self.matmul.bias.contiguous()
        self._ready = True

    @staticmethod
    def _split(x):
        numel = x.numel()
        hi = torch.empty(x.shape, device=x.device, dtype=torch.float16)
        lo = torch.empty(x.shape, device=x.device, dtype=torch.float16)
        BLOCK = 8192
        grid = (triton.cdiv(numel, BLOCK),)
        _split_kernel[grid](x.reshape(-1), hi.reshape(-1), lo.reshape(-1), numel,
                            BLOCK=BLOCK, num_warps=8)
        return hi, lo

    def forward(self, x):
        if not self._ready:
            self._prepare()
        M, K = x.shape
        N = self._w_hi_t.shape[1]
        x = x.contiguous()
        x_hi, x_lo = self._split(x)
        out = torch.empty((M, N), device=x.device, dtype=torch.float32)
        grid = (triton.cdiv(M, _BM) * triton.cdiv(N, _BN),)
        _gemm_fp16x3_tma[grid](x_hi, x_lo, self._w_hi_t, self._w_lo_t,
                               self._bias, out, M, N, K,
                               BM=_BM, BN=_BN, BK=_BK, GROUP_M=_GM,
                               num_warps=8, num_stages=3)
        return out
