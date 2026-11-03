import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _gemm3xfp16(X, WH, WL, B, Y, M, N, K,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr):
    pid = tl.program_id(0)
    grid_m = M // BM
    grid_n = N // BN
    width = GROUP * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP, GROUP)
    pid_m = group_id * GROUP + (pid % group_size)
    pid_n = (pid % width) // group_size
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    Xp = X + rm[:, None] * K + rk[None, :]
    WHp = WH + rn[None, :] * K + rk[:, None]
    WLp = WL + rn[None, :] * K + rk[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K, BK):
        x = tl.load(Xp)
        xh = x.to(tl.float16)
        xl = (x - xh.to(tl.float32)).to(tl.float16)
        wh = tl.load(WHp)
        wl = tl.load(WLp)
        acc = tl.dot(xh, wh, acc)
        acc = tl.dot(xl, wh, acc)
        acc = tl.dot(xh, wl, acc)
        Xp += BK
        WHp += BK
        WLp += BK
    b = tl.load(B + rn)
    acc = acc + b[None, :]
    Yp = Y + rm[:, None] * N + rn[None, :]
    tl.store(Yp, acc)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self._wh = None
        self._wl = None
        self._b = None

    def _prepare(self, device):
        with torch.no_grad():
            s = 1.0 + float(self.scaling_factor)
            w = self.matmul.weight.detach().to(device=device, dtype=torch.float32) * s
            wh = w.half().contiguous()
            wl = (w - wh.float()).half().contiguous()
            b = (self.matmul.bias.detach().to(device=device, dtype=torch.float32) * s).contiguous()
        self._wh, self._wl, self._b = wh, wl, b

    def forward(self, x):
        x = x.contiguous()
        if self._wh is None or self._wh.device != x.device:
            self._prepare(x.device)
        M, K = x.shape
        N = self._wh.shape[0]
        if M % 128 or N % 128 or K % 64:
            with torch.no_grad():
                w = self._wh.float() + self._wl.float()
                return F.linear(x, w, self._b)
        y = torch.empty((M, N), device=x.device, dtype=torch.float32)
        grid = ((M // 128) * (N // 128),)
        _gemm3xfp16[grid](x, self._wh, self._wl, self._b, y, M, N, K,
                          BM=128, BN=128, BK=64, GROUP=8,
                          num_warps=8, num_stages=3)
        return y
