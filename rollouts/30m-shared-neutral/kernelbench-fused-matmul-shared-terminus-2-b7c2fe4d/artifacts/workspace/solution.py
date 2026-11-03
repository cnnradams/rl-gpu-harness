import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_split_mm(
    x_ptr, bh_ptr, bl_ptr, bias_ptr, out_ptr,
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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < N

    x_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
    bh_ptrs = bh_ptr + offs_k[:, None] * N + offs_n[None, :]
    bl_ptrs = bl_ptr + offs_k[:, None] * N + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        ka = offs_k[None, :] < k_rem
        kb = offs_k[:, None] < k_rem
        x = tl.load(x_ptrs, mask=m_mask[:, None] & ka, other=0.0)
        xh = x.to(tl.float16)
        xl = (x - xh.to(tl.float32)).to(tl.float16)
        bh = tl.load(bh_ptrs, mask=kb & n_mask[None, :], other=0.0)
        bl = tl.load(bl_ptrs, mask=kb & n_mask[None, :], other=0.0)
        acc = tl.dot(xh, bl, acc)
        acc = tl.dot(xl, bh, acc)
        acc = tl.dot(xh, bh, acc)
        x_ptrs += BLOCK_K
        bh_ptrs += BLOCK_K * N
        bl_ptrs += BLOCK_K * N

    bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        lin = nn.Linear(in_features, out_features)
        with torch.no_grad():
            s = 1.0 + float(scaling_factor)
            w = (lin.weight.detach() * s).t().contiguous()
            b = (lin.bias.detach() * s).contiguous()
            wh = w.half()
            wl = (w - wh.float()).half()
        self.register_buffer("bh_i16", wh.view(torch.int16))
        self.register_buffer("bl_i16", wl.view(torch.int16))
        self.register_buffer("bias", b)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous()
        M, K = x.shape
        N = self.out_features
        bh = self.bh_i16.view(torch.float16)
        bl = self.bl_i16.view(torch.float16)
        out = torch.empty((M, N), device=x.device, dtype=torch.float32)
        BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 256, 128, 32, 8
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        _fused_split_mm[grid](
            x, bh, bl, self.bias, out,
            M, N, K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
            num_warps=8, num_stages=4,
        )
        return out
