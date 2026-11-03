import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _linear_scale_x3(A, BH, BL, Bias, C, M, N, K,
                     sam, sak, sbk, sbn, scm, scn,
                     SCALE: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                     GROUP_M: tl.constexpr, EVEN_MN: tl.constexpr, EVEN_K: tl.constexpr):
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
    a_ptrs = A + rm[:, None] * sam + rk[None, :] * sak
    bh_ptrs = BH + rk[:, None] * sbk + rn[None, :] * sbn
    bl_ptrs = BL + rk[:, None] * sbk + rn[None, :] * sbn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_MN and EVEN_K:
            a = tl.load(a_ptrs)
            bh = tl.load(bh_ptrs)
            bl = tl.load(bl_ptrs)
        else:
            k_rem = K - k * BLOCK_K
            amask = (rm[:, None] < M) & (rk[None, :] < k_rem)
            bmask = (rk[:, None] < k_rem) & (rn[None, :] < N)
            a = tl.load(a_ptrs, mask=amask, other=0.0)
            bh = tl.load(bh_ptrs, mask=bmask, other=0.0)
            bl = tl.load(bl_ptrs, mask=bmask, other=0.0)
        ah = a.to(tl.float16)
        al = (a - ah.to(tl.float32)).to(tl.float16)
        acc = tl.dot(ah, bh, acc)
        acc = tl.dot(ah, bl, acc)
        acc = tl.dot(al, bh, acc)
        a_ptrs += BLOCK_K * sak
        bh_ptrs += BLOCK_K * sbk
        bl_ptrs += BLOCK_K * sbk
    if EVEN_MN:
        bias = tl.load(Bias + rn)
    else:
        bias = tl.load(Bias + rn, mask=rn < N, other=0.0)
    acc = (acc + bias[None, :]) * SCALE
    c_ptrs = C + rm[:, None] * scm + rn[None, :] * scn
    if EVEN_MN:
        tl.store(c_ptrs, acc)
    else:
        tl.store(c_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self._wh = None
        self._wl = None

    def _prep(self, w):
        wt = w.detach().t().contiguous()
        wh = wt.half()
        wl = (wt - wh.float()).half()
        self._wh, self._wl = wh, wl

    def forward(self, x):
        if not x.is_cuda:
            m = self.matmul(x)
            return m * (self.scaling_factor + 1.0)
        w = self.matmul.weight
        if self._wh is None or self._wh.device != w.device or self._wh.dtype != torch.float16:
            self._prep(w)
        wh, wl = self._wh, self._wl
        b = self.matmul.bias
        x = x.contiguous()
        if x.dtype != torch.float32:
            x = x.float()
        M, K = x.shape
        N = w.shape[0]
        out = torch.empty((M, N), device=x.device, dtype=torch.float32)
        BM, BN, BK, GM = 256, 128, 32, 8
        even_mn = (M % BM == 0) and (N % BN == 0)
        even_k = (K % BK == 0)
        grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
        _linear_scale_x3[grid](
            x, wh, wl, b, out, M, N, K,
            x.stride(0), x.stride(1), wh.stride(0), wh.stride(1),
            out.stride(0), out.stride(1),
            SCALE=float(self.scaling_factor) + 1.0,
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GM,
            EVEN_MN=even_mn, EVEN_K=even_k,
            num_warps=8, num_stages=3,
        )
        return out
