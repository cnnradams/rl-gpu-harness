import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:
    HAS_TRITON = False

if HAS_TRITON:
    @triton.autotune(
        configs=[
            triton.Config({'BM': 256, 'BN': 128, 'BK': 64, 'GM': 8}, num_warps=8, num_stages=3),
            triton.Config({'BM': 256, 'BN': 128, 'BK': 32, 'GM': 8}, num_warps=8, num_stages=4),
            triton.Config({'BM': 128, 'BN': 128, 'BK': 64, 'GM': 8}, num_warps=8, num_stages=4),
            triton.Config({'BM': 128, 'BN': 256, 'BK': 64, 'GM': 8}, num_warps=8, num_stages=2),
            triton.Config({'BM': 64, 'BN': 256, 'BK': 64, 'GM': 8}, num_warps=4, num_stages=4),
        ],
        key=['M', 'N', 'K'],
    )
    @triton.jit
    def _wsplit_kernel(XP, WHP, WLP, BP, YP, M, N, K, s,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                       GM: tl.constexpr):
        pid = tl.program_id(0)
        grid_m = tl.cdiv(M, BM)
        grid_n = tl.cdiv(N, BN)
        width = GM * grid_n
        group_id = pid // width
        group_size = tl.minimum(grid_m - group_id * GM, GM)
        pid_m = group_id * GM + (pid % group_size)
        pid_n = (pid % width) // group_size

        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)

        xptrs = XP + rm[:, None] * K + rk[None, :]
        whptrs = WHP + rk[:, None] * N + rn[None, :]
        wlptrs = WLP + rk[:, None] * N + rn[None, :]

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _k in range(0, K, BK):
            xv = tl.load(xptrs)
            xh = xv.to(tl.float16)
            xl = (xv - xh.to(tl.float32)).to(tl.float16)
            wh = tl.load(whptrs)
            wl = tl.load(wlptrs)
            acc = tl.dot(xh, wh, acc)
            acc = tl.dot(xh, wl, acc)
            acc = tl.dot(xl, wh, acc)
            xptrs += BK
            whptrs += BK * N
            wlptrs += BK * N

        bv = tl.load(BP + rn).to(tl.float32)
        acc = acc * s + bv[None, :]
        yptrs = YP + rm[:, None] * N + rn[None, :]
        tl.store(yptrs, acc)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.alpha = 1.0 + scaling_factor
        self._wh = None
        self._wl = None
        self._bs = None
        self._wkey = None

    def _fallback(self, x):
        return torch.addmm(self.matmul.bias, x, self.matmul.weight.t(),
                           beta=self.alpha, alpha=self.alpha)

    def forward(self, x):
        w = self.matmul.weight
        b = self.matmul.bias
        if (not HAS_TRITON) or (not x.is_cuda) or x.dtype != torch.float32 or x.dim() != 2:
            return self._fallback(x)
        M, K = x.shape
        N, Kw = w.shape
        if Kw != K or (M % 64) or (N % 64) or (K % 32):
            return self._fallback(x)
        key = (w.data.data_ptr(), b.data.data_ptr(), w.data.device, w.data._version)
        if self._wkey != key:
            wt = w.data.t().contiguous()
            wh = wt.half()
            wl = (wt - wh.float()).half()
            self._wh = wh.contiguous()
            self._wl = wl.contiguous()
            self._bs = (b.data * self.alpha).contiguous()
            self._wkey = key
        x = x.contiguous()
        y = torch.empty((M, N), device=x.device, dtype=torch.float32)
        grid = lambda meta: (triton.cdiv(M, meta['BM']) * triton.cdiv(N, meta['BN']),)
        _wsplit_kernel[grid](
            x, self._wh, self._wl, self._bs, y, M, N, K, float(self.alpha),
        )
        return y
