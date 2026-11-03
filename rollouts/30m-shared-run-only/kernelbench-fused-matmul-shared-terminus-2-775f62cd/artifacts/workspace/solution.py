import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


if _HAS_TRITON:
    def _cfgs():
        cfgs = []
        for bm, bn, bk, ns, nw in [
            (256, 128, 32, 4, 8),
            (128, 128, 64, 3, 8),
            (128, 128, 32, 5, 8),
        ]:
            cfgs.append(
                triton.Config(
                    {"BM": bm, "BN": bn, "BK": bk, "GROUP": 8},
                    num_warps=nw, num_stages=ns,
                )
            )
        return cfgs

    @triton.autotune(configs=_cfgs(), key=["M", "N", "K"])
    @triton.jit
    def _split_gemm_kernel(
        X, WH, WL, B, Y, M, N, K, s,
        stride_xm, stride_xk,
        stride_wk, stride_wn,
        stride_ym, stride_yn,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr,
    ):
        pid = tl.program_id(0)
        grid_m = tl.cdiv(M, BM)
        grid_n = tl.cdiv(N, BN)
        width = GROUP * grid_n
        group_id = pid // width
        group_size = tl.minimum(grid_m - group_id * GROUP, GROUP)
        pid_m = group_id * GROUP + (pid % group_size)
        pid_n = (pid % width) // group_size

        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        X_ptr = X + rm[:, None] * stride_xm + rk[None, :] * stride_xk
        WH_ptr = WH + rk[:, None] * stride_wk + rn[None, :] * stride_wn
        WL_ptr = WL + rk[:, None] * stride_wk + rn[None, :] * stride_wn
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, K, BK):
            x = tl.load(X_ptr)
            xh = x.to(tl.float16)
            xl = (x - xh.to(tl.float32)).to(tl.float16)
            wh = tl.load(WH_ptr)
            wl = tl.load(WL_ptr)
            acc = tl.dot(xh, wh, acc)
            acc = tl.dot(xl, wh, acc)
            acc = tl.dot(xh, wl, acc)
            X_ptr += BK * stride_xk
            WH_ptr += BK * stride_wk
            WL_ptr += BK * stride_wk
        b = tl.load(B + rn).to(tl.float32)
        y = acc * s + b * s
        Y_ptr = Y + rm[:, None] * stride_ym + rn[None, :] * stride_yn
        tl.store(Y_ptr, y)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self._cache_key = None
        self._wh = None
        self._wl = None
        self._b = None
        self._s = None

    def _prepare(self, x):
        w = self.matmul.weight
        key = (w.data_ptr(), w._version, str(w.device), x.shape)
        if self._cache_key == key:
            return
        with torch.no_grad():
            wt = w.detach().t().contiguous()  # [K, N] fp32
            wh = wt.to(torch.float16)
            wl = (wt - wh.to(torch.float32)).to(torch.float16)
            self._wh = wh.contiguous()
            self._wl = wl.contiguous()
            self._b = self.matmul.bias.detach().clone()
            self._s = float(self.scaling_factor) + 1.0
            self._cache_key = key

    def forward(self, x):
        if (
            _HAS_TRITON
            and x.device.type == "cuda"
            and x.dtype == torch.float32
            and x.dim() == 2
        ):
            if not x.is_contiguous():
                x = x.contiguous()
            self._prepare(x)
            M, K = x.shape
            N = self._wh.shape[1]
            if M % 64 == 0 and N % 64 == 0 and K % 32 == 0:
                y = torch.empty((M, N), device=x.device, dtype=torch.float32)
                grid = lambda meta: (
                    triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"]),
                )
                _split_gemm_kernel[grid](
                    x, self._wh, self._wl, self._b, y,
                    M, N, K, self._s,
                    x.stride(0), x.stride(1),
                    self._wh.stride(0), self._wh.stride(1),
                    y.stride(0), y.stride(1),
                )
                return y
        y = self.matmul(x)
        return y * self.scaling_factor + y.detach()
