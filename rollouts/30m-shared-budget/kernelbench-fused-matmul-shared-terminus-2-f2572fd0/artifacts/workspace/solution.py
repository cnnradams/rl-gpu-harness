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
    def _split_gemm_kernel(
        A, BH, BL, Bias, C,
        M, N, K,
        sam, sak,
        sbk, sbn,
        scm, scn,
        scale,
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

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)

        A_ptr = A + rm[:, None] * sam + rk[None, :] * sak
        BH_ptr = BH + rk[:, None] * sbk + rn[None, :] * sbn
        BL_ptr = BL + rk[:, None] * sbk + rn[None, :] * sbn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(A_ptr)
            ah = a.to(tl.float16)
            al = (a - ah.to(tl.float32)).to(tl.float16)
            bh = tl.load(BH_ptr)
            bl = tl.load(BL_ptr)
            acc = tl.dot(ah, bh, acc)
            acc = tl.dot(al, bh, acc)
            acc = tl.dot(ah, bl, acc)
            A_ptr += BLOCK_K * sak
            BH_ptr += BLOCK_K * sbk
            BL_ptr += BLOCK_K * sbk

        bias = tl.load(Bias + rn)
        acc = acc * scale + bias[None, :]
        C_ptr = C + rm[:, None] * scm + rn[None, :] * scn
        tl.store(C_ptr, acc)


class ModelNew(nn.Module):
    """Linear + scale + residual implemented as (1+s) * (Wx + b) using a
    Markidis-style split-fp16 tensor-core GEMM with fp32 accumulation."""

    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self._scale1p = float(1.0 + scaling_factor)
        self._key = None
        self._wh = None
        self._wl = None
        self._bias = None

    def _prepare(self, w, b):
        key = (w.data_ptr(), w.device, w._version, b.data_ptr() if b is not None else None)
        if self._key == key:
            return
        wd = w.detach()
        w16 = wd.to(torch.float16)
        wl16 = (wd - w16.to(torch.float32)).to(torch.float16)
        self._wh = w16.contiguous()
        self._wl = wl16.contiguous()
        if b is not None:
            self._bias = (b.detach() * self._scale1p).contiguous()
        else:
            self._bias = torch.zeros(wd.shape[0], device=wd.device, dtype=torch.float32)
        self._key = key

    def forward(self, x):
        w = self.matmul.weight
        b = self.matmul.bias
        if (not _HAS_TRITON) or (not x.is_cuda) or x.dtype != torch.float32 or x.dim() != 2:
            y = F.linear(x, w, b)
            return y * self.scaling_factor + y
        M, K = x.shape
        N = w.shape[0]
        BM, BN, BK = 128, 128, 64
        if (M % BM) != 0 or (N % BN) != 0 or (K % BK) != 0:
            y = F.linear(x, w, b)
            return y * self.scaling_factor + y
        self._prepare(w, b)
        x = x.contiguous()
        out = torch.empty((M, N), device=x.device, dtype=torch.float32)
        wh = self._wh
        wl = self._wl
        grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
        _split_gemm_kernel[grid](
            x, wh, wl, self._bias, out,
            M, N, K,
            x.stride(0), x.stride(1),
            wh.stride(1), wh.stride(0),
            out.stride(0), out.stride(1),
            self._scale1p,
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=8,
            num_warps=8, num_stages=3,
        )
        return out
