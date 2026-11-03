import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False

if _HAS_TRITON:
    @triton.jit
    def _mm3_kernel(X, WH, WL, BIAS, Y, M, N, K,
                    sxm, sxk, swk, swn, sym, syn,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                    GROUP_M: tl.constexpr, SCALE: tl.constexpr):
        pid = tl.program_id(0)
        grid_m = tl.cdiv(M, BM)
        grid_n = tl.cdiv(N, BN)
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // group_size
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        Xp = X + rm[:, None] * sxm + rk[None, :] * sxk
        WHp = WH + rk[:, None] * swk + rn[None, :] * swn
        WLp = WL + rk[:, None] * swk + rn[None, :] * swn
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            a = tl.load(Xp)
            ah = a.to(tl.bfloat16)
            al = (a - ah.to(tl.float32)).to(tl.bfloat16)
            bh = tl.load(WHp)
            bl = tl.load(WLp)
            acc = tl.dot(ah, bh, acc)
            acc = tl.dot(al, bh, acc)
            acc = tl.dot(ah, bl, acc)
            Xp += BK * sxk
            WHp += BK * swk
            WLp += BK * swk
        bias = tl.load(BIAS + rn)
        y = SCALE * acc + SCALE * bias[None, :]
        Yp = Y + rm[:, None] * sym + rn[None, :] * syn
        tl.store(Yp, y)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self._split_cache = None
        self._ok = None

    def _get_split(self, device):
        c = self._split_cache
        if c is not None and c[0].device == device:
            return c
        W = self.matmul.weight.data
        Wt = W.t().contiguous()
        wh = Wt.bfloat16()
        wl = (Wt - wh.float()).bfloat16()
        b = self.matmul.bias.data.contiguous()
        self._split_cache = (wh, wl, b)
        return self._split_cache

    def forward(self, x):
        if _HAS_TRITON and x.is_cuda and x.dtype == torch.float32 and x.dim() == 2:
            try:
                if self._ok is False:
                    raise RuntimeError("triton path disabled")
                M, K = x.shape
                N = self.matmul.weight.shape[0]
                if K % 32 == 0 and M % 128 == 0 and N % 128 == 0:
                    wh, wl, b = self._get_split(x.device)
                    Y = torch.empty(M, N, device=x.device, dtype=torch.float32)
                    BM, BN, BK, GM = 128, 128, 32, 64
                    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
                    _mm3_kernel[grid](
                        x, wh, wl, b, Y, M, N, K,
                        x.stride(0), x.stride(1),
                        wh.stride(0), wh.stride(1),
                        Y.stride(0), Y.stride(1),
                        BM, BN, BK, GM,
                        float(self.scaling_factor) + 1.0,
                        num_warps=8, num_stages=4,
                    )
                    self._ok = True
                    return Y
            except Exception:
                self._ok = False
        m = self.matmul(x)
        return m * self.scaling_factor + m
