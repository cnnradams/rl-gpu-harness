import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _split_fp32(X, HI, LO, numel, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel
        v = tl.load(X + offs, mask=mask)
        h = v.to(tl.float16)
        l = (v - h.to(tl.float32)).to(tl.float16)
        tl.store(HI + offs, h, mask=mask)
        tl.store(LO + offs, l, mask=mask)

    @triton.jit
    def _gemm_fp16x3(AH, AL, WH, WL, BIAS, C, M, N, K, scale,
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
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        ah_ptrs = AH + offs_m[:, None] * K + offs_k[None, :]
        al_ptrs = AL + offs_m[:, None] * K + offs_k[None, :]
        wh_ptrs = WH + offs_k[:, None] * N + offs_n[None, :]
        wl_ptrs = WL + offs_k[:, None] * N + offs_n[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BK)):
            ahi = tl.load(ah_ptrs)
            alo = tl.load(al_ptrs)
            bhi = tl.load(wh_ptrs)
            blo = tl.load(wl_ptrs)
            acc = tl.dot(ahi, bhi, acc)
            acc = tl.dot(ahi, blo, acc)
            acc = tl.dot(alo, bhi, acc)
            ah_ptrs += BK
            al_ptrs += BK
            wh_ptrs += BK * N
            wl_ptrs += BK * N
        bv = tl.load(BIAS + offs_n)
        c = acc * scale + bv[None, :]
        tl.store(C + offs_m[:, None] * N + offs_n[None, :], c)


_BM, _BN, _BK, _GM = 128, 256, 32, 16


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self._w_hi = None
        self._w_lo = None
        self._bias_s = None

    def _prepare(self, x):
        w = self.matmul.weight
        b = self.matmul.bias
        s = float(self.scaling_factor)
        if (self._w_hi is None or self._w_hi.device != x.device):
            with torch.no_grad():
                w_t = w.detach().t().contiguous()
                w_hi = w_t.half()
                w_lo = (w_t - w_hi.float()).half()
                self._w_hi = w_hi
                self._w_lo = w_lo
                self._bias_s = (b.detach() * (1.0 + s)).contiguous()
        return self._w_hi, self._w_lo, self._bias_s

    def forward(self, x):
        s = float(self.scaling_factor)
        if not (_HAS_TRITON and x.is_cuda and x.dtype == torch.float32 and x.dim() == 2):
            y = torch.nn.functional.linear(x, self.matmul.weight, self.matmul.bias)
            return y * s + y
        M, K = x.shape
        N = self.matmul.weight.shape[0]
        if M % _BM != 0 or N % _BN != 0 or K % _BK != 0:
            y = torch.nn.functional.linear(x, self.matmul.weight, self.matmul.bias)
            return y * s + y
        w_hi, w_lo, bias_s = self._prepare(x)
        x = x.contiguous()
        numel = M * K
        x_hi = torch.empty((M, K), device=x.device, dtype=torch.float16)
        x_lo = torch.empty((M, K), device=x.device, dtype=torch.float16)
        SB = 4096
        _split_fp32[(triton.cdiv(numel, SB),)](x, x_hi, x_lo, numel, SB, num_warps=8)
        out = torch.empty((M, N), device=x.device, dtype=torch.float32)
        grid = (triton.cdiv(M, _BM) * triton.cdiv(N, _BN),)
        _gemm_fp16x3[grid](x_hi, x_lo, w_hi, w_lo, bias_s, out, M, N, K,
                           1.0 + s, _BM, _BN, _BK, _GM, num_warps=8, num_stages=3)
        return out
