import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _lin15_kernel(X, WH, WL, B, O, M, N, K,
                  sxm, sxk, swk, swn, slk, sln, som, son,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BM)
    grid_n = tl.cdiv(N, BN)
    width = GM * grid_n
    gid = pid // width
    gsize = tl.minimum(grid_m - gid * GM, GM)
    pid_m = gid * GM + (pid % gsize)
    pid_n = (pid % width) // gsize
    om = pid_m * BM + tl.arange(0, BM)
    on = pid_n * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)
    xp = X + om[:, None] * sxm + ok[None, :] * sxk
    wp = WH + ok[:, None] * swk + on[None, :] * swn
    lp = WL + ok[:, None] * slk + on[None, :] * sln
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(0, K, BK):
        x = tl.load(xp)
        wh = tl.load(wp)
        wl = tl.load(lp)
        xh = x.to(tl.float16)
        xl = (x - xh.to(tl.float32)).to(tl.float16)
        acc = tl.dot(xh, wh, acc)
        acc = tl.dot(xl, wh, acc)
        acc = tl.dot(xh, wl, acc)
        xp += BK * sxk
        wp += BK * swk
        lp += BK * slk
    b = tl.load(B + on)
    acc = acc + b[None, :].to(tl.float32)
    op = O + om[:, None] * som + on[None, :] * son
    tl.store(op, acc)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self._wh = None
        self._wl = None
        self._b = None

    def _prepare(self):
        with torch.no_grad():
            s = 1.0 + float(self.scaling_factor)
            w = self.matmul.weight
            wt = (w * s).t().contiguous()
            wh = wt.half()
            wl = (wt - wh.float()).half()
            self._wh = wh
            self._wl = wl
            self._b = (self.matmul.bias * s).contiguous()

    def forward(self, x):
        if self._wh is None or self._wh.device != x.device:
            self._prepare()
        M, K = x.shape
        N = self._wh.shape[1]
        BM, BN, BK, GM = 256, 128, 32, 8
        if x.is_contiguous() and M % BM == 0 and N % BN == 0 and K % BK == 0 and x.dtype == torch.float32:
            out = torch.empty((M, N), device=x.device, dtype=torch.float32)
            grid = ((M // BM) * (N // BN),)
            _lin15_kernel[grid](
                x, self._wh, self._wl, self._b, out, M, N, K,
                x.stride(0), x.stride(1),
                self._wh.stride(0), self._wh.stride(1),
                self._wl.stride(0), self._wl.stride(1),
                out.stride(0), out.stride(1),
                BM=BM, BN=BN, BK=BK, GM=GM,
                num_warps=8, num_stages=4,
            )
            return out
        t = F.linear(x, self.matmul.weight, self.matmul.bias)
        return t * self.scaling_factor + t
