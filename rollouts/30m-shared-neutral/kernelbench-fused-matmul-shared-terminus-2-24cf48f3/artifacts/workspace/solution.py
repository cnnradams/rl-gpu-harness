import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

try:
    from triton.tools.tensor_descriptor import TensorDescriptor
    _HAS_TD = True
except Exception:
    _HAS_TD = False


@triton.jit
def _split_cat_kernel(x_ptr, a3_ptr, M, K, sa, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < K
    x = tl.load(x_ptr + row * K + offs, mask=mask)
    hi = x.to(tl.float16)
    lo = (x - hi.to(tl.float32)).to(tl.float16)
    base = a3_ptr + row * sa
    tl.store(base + offs, hi, mask=mask)
    tl.store(base + K + offs, hi, mask=mask)
    tl.store(base + 2 * K + offs, lo, mask=mask)


@triton.jit
def _gemm_tma(
    a_desc, b_desc, bias_ptr, c_desc,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = a_desc.load([off_m, k * BLOCK_K])
        b = b_desc.load([k * BLOCK_K, off_n])
        acc = tl.dot(a, b, acc)
    rn = off_n + tl.arange(0, BLOCK_N)
    bias = tl.load(bias_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    c_desc.store([off_m, off_n], acc.to(tl.float32))


@triton.jit
def _gemm_masked(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    sam, sak, sbk, sbn, scm, scn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + rm[:, None] * sam + rk[None, :] * sak
    b_ptrs = b_ptr + rk[:, None] * sbk + rn[None, :] * sbn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            krem = K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=rk[None, :] < krem, other=0.0)
            b = tl.load(b_ptrs, mask=rk[:, None] < krem, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * sak
        b_ptrs += BLOCK_K * sbk
    bias = tl.load(bias_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    c_ptrs = c_ptr + rm[:, None] * scm + rn[None, :] * scn
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        lin = nn.Linear(in_features, out_features)
        s = 1.0 + float(scaling_factor)
        with torch.no_grad():
            lin.weight.mul_(s)
            lin.bias.mul_(s)
        self.weight = lin.weight
        self.bias = lin.bias
        self._b3 = None

    def _prep(self, device):
        w = self.weight.detach()
        w_hi = w.half()
        w_lo = (w - w_hi.float()).half()
        self._b3 = torch.cat([w_hi.t(), w_lo.t(), w_hi.t()], dim=0).contiguous()

    def forward(self, x):
        if x.is_cuda and x.dtype == torch.float32 and x.dim() == 2 and x.is_contiguous():
            M, K = x.shape
            N = self.weight.shape[0]
            if self._b3 is None or self._b3.device != x.device:
                self._prep(x.device)
            b3 = self._b3
            bias = self.bias
            K3 = 3 * K
            a3 = torch.empty((M, K3), device=x.device, dtype=torch.float16)
            grid_split = (triton.cdiv(K, 2048), M)
            _split_cat_kernel[grid_split](x, a3, M, K, a3.stride(0), BLOCK=2048, num_warps=8)
            out = torch.empty((M, N), device=x.device, dtype=torch.float32)
            if _HAS_TD and M % 128 == 0 and K % 64 == 0 and N % 256 == 0:
                bm, bn, bk, gm, nw, ns = 128, 256, 64, 8, 8, 4
                a_desc = TensorDescriptor(a3, a3.shape, a3.stride(), [bm, bk])
                b_desc = TensorDescriptor(b3, b3.shape, b3.stride(), [bk, bn])
                c_desc = TensorDescriptor(out, out.shape, out.stride(), [bm, bn])
                grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
                _gemm_tma[grid](a_desc, b_desc, bias, c_desc, M, N, K3,
                                BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=gm,
                                num_warps=nw, num_stages=ns)
            else:
                bm, bn, bk, gm, nw, ns = 128, 128, 64, 8, 8, 3
                grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
                _gemm_masked[grid](a3, b3, bias, out, M, N, K3,
                                   a3.stride(0), a3.stride(1),
                                   b3.stride(0), b3.stride(1),
                                   out.stride(0), out.stride(1),
                                   BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=gm,
                                   EVEN_K=(K3 % bk == 0),
                                   num_warps=nw, num_stages=ns)
            return out
        return F.linear(x, self.weight, self.bias)
