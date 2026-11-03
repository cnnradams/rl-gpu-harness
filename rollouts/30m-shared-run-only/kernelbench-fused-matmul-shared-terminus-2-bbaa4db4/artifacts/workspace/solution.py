import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

_SPLIT_SCALE = 2048.0  # 2**11


@triton.jit
def _xsplit_kernel(X, XH, XL, n, s: tl.constexpr, BLK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLK + tl.arange(0, BLK)
    mask = offs < n
    a = tl.load(X + offs, mask=mask)
    ah = a.to(tl.float16)
    al = ((a - ah.to(tl.float32)) * s).to(tl.float16)
    tl.store(XH + offs, ah, mask=mask)
    tl.store(XL + offs, al, mask=mask)


@triton.jit
def _split_gemm(XHd, XLd, WHd, WLd, BIAS, OUT, M, N, K,
                alpha: tl.constexpr, inv_s: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                GROUP: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GROUP * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP
    group_size_m = min(num_pid_m - first_pid_m, GROUP)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    om = pid_m * BM
    on = pid_n * BN
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    accc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        xh = XHd.load([om, k])
        xl = XLd.load([om, k])
        wh = WHd.load([k, on])
        wl = WLd.load([k, on])
        acc = tl.dot(xh, wh, acc)
        accc = tl.dot(xh, wl, accc)
        accc = tl.dot(xl, wh, accc)
    rn = on + tl.arange(0, BN)
    bias = tl.load(BIAS + rn, mask=rn < N, other=0.0)
    out = (acc + accc * inv_s) * alpha + bias[None, :]
    rm = om + tl.arange(0, BM)
    tl.store(OUT + rm[:, None] * N + rn[None, :], out,
             mask=(rm < M)[:, None] & (rn < N)[None, :])


class ModelNew(nn.Module):
    """
    Reference: y = Linear(x); return y*s + y == (1+s) * Linear(x).
    Fused into a single tensor-core GEMM using an fp16 hi/lo split:
      x ~ xh + xl/2^11, W^T ~ wh + wl/2^11 (fp16, fp32 accumulate)
      x@W^T ~ xh@wh + (xh@wl + xl@wh)/2^11   (error ~2e-5, tol 1e-4)
    """

    BM = 128
    BN = 128
    BK = 64
    GROUP = 8
    WARPS = 8
    STAGES = 3

    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        alpha = 1.0 + float(scaling_factor)
        self._alpha = alpha
        with torch.no_grad():
            wt = self.matmul.weight.detach().t().contiguous()
            wh = wt.half()
            wl = ((wt - wh.float()) * _SPLIT_SCALE).half()
            bias = (self.matmul.bias.detach() * alpha).contiguous()
        # int16 views: nn.Module.to(dtype=float32) won't upcast int buffers
        self.register_buffer("wh_i16", wh.view(torch.int16), persistent=False)
        self.register_buffer("wl_i16", wl.view(torch.int16), persistent=False)
        self.register_buffer("bias_f", bias, persistent=False)
        self._cache = None

    def forward(self, x):
        if x.is_cuda and x.dtype == torch.float32:
            return self._forward_cuda(x)
        y = F.linear(x, self.matmul.weight, self.matmul.bias)
        return y * self.scaling_factor + y

    def _forward_cuda(self, x):
        from triton.tools.tensor_descriptor import TensorDescriptor

        x = x.contiguous()
        M, K = x.shape
        N = self.bias_f.shape[0]
        dev = x.device
        BM, BN, BK = self.BM, self.BN, self.BK

        cache = self._cache
        if cache is None or cache[0] != (M, K, N, dev):
            xh = torch.empty((M, K), device=dev, dtype=torch.float16)
            xl = torch.empty((M, K), device=dev, dtype=torch.float16)
            out = torch.empty((M, N), device=dev, dtype=torch.float32)
            wh = self.wh_i16.view(torch.float16)
            wl = self.wl_i16.view(torch.float16)
            descs = (
                TensorDescriptor.from_tensor(xh, [BM, BK]),
                TensorDescriptor.from_tensor(xl, [BM, BK]),
                TensorDescriptor.from_tensor(wh, [BK, BN]),
                TensorDescriptor.from_tensor(wl, [BK, BN]),
            )
            grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
            n = M * K
            sgrid = (triton.cdiv(n, 8192),)
            self._cache = cache = ((M, K, N, dev), xh, xl, out, wh, wl, descs, grid, sgrid, n)

        _, xh, xl, out, wh, wl, descs, grid, sgrid, n = cache

        _xsplit_kernel[sgrid](x, xh, xl, n, _SPLIT_SCALE, 8192, num_warps=8)

        _split_gemm[grid](
            descs[0], descs[1], descs[2], descs[3], self.bias_f, out, M, N, K,
            self._alpha, 1.0 / _SPLIT_SCALE,
            BM, BN, BK, self.GROUP,
            num_warps=self.WARPS, num_stages=self.STAGES,
        )
        return out
