import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _split_fp16_kernel(x_ptr, x1_ptr, x2_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    v = tl.load(x_ptr + offs, mask=mask, other=0.0)
    v1 = v.to(tl.float16)
    v2 = (v - v1.to(tl.float32)).to(tl.float16)
    tl.store(x1_ptr + offs, v1, mask=mask)
    tl.store(x2_ptr + offs, v2, mask=mask)


@triton.jit
def _split_gemm_kernel(x1p, x2p, w1p, w2p, bp, op, M, N, K,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                       GROUP: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BM)
    grid_n = tl.cdiv(N, BN)
    width = GROUP * grid_n
    gid = pid // width
    gsize = tl.minimum(grid_m - gid * GROUP, GROUP)
    pid_m = gid * GROUP + (pid % gsize)
    pid_n = (pid % width) // gsize
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    x1p += rm[:, None] * K + rk[None, :]
    x2p += rm[:, None] * K + rk[None, :]
    w1p += rk[:, None] * N + rn[None, :]
    w2p += rk[:, None] * N + rn[None, :]
    a1 = tl.zeros((BM, BN), dtype=tl.float32)
    a2 = tl.zeros((BM, BN), dtype=tl.float32)
    a3 = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        if EVEN:
            va1 = tl.load(x1p)
            va2 = tl.load(x2p)
            vb1 = tl.load(w1p)
            vb2 = tl.load(w2p)
        else:
            k_mask = rk[None, :] < K - k * BK
            kb_mask = (rk[:, None] < K - k * BK) & (rn[None, :] < N)
            va1 = tl.load(x1p, mask=(rm[:, None] < M) & k_mask, other=0.0)
            va2 = tl.load(x2p, mask=(rm[:, None] < M) & k_mask, other=0.0)
            vb1 = tl.load(w1p, mask=kb_mask, other=0.0)
            vb2 = tl.load(w2p, mask=kb_mask, other=0.0)
        a1 = tl.dot(va1, vb1, a1)
        a2 = tl.dot(va2, vb1, a2)
        a3 = tl.dot(va1, vb2, a3)
        x1p += BK
        x2p += BK
        w1p += BK * N
        w2p += BK * N
    acc = a1 + a2 + a3
    if EVEN:
        bias = tl.load(bp + rn)
        acc = acc + bias[None, :]
        tl.store(op + rm[:, None] * N + rn[None, :], acc)
    else:
        n_mask = rn < N
        bias = tl.load(bp + rn, mask=n_mask, other=0.0)
        acc = acc + bias[None, :]
        tl.store(op + rm[:, None] * N + rn[None, :], acc,
                 mask=(rm[:, None] < M) & n_mask[None, :])


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self._prepared = False

    def _prepare(self):
        with torch.no_grad():
            s = 1.0 + float(self.scaling_factor)
            w = (self.matmul.weight.detach().float() * s).contiguous()
            b = (self.matmul.bias.detach().float() * s).contiguous()
            w1 = w.half()
            w2 = (w - w1.float()).half()
            self._w1t = w1.t().contiguous()
            self._w2t = w2.t().contiguous()
            self._bias = b
        self._prepared = True

    def forward(self, x):
        if not x.is_cuda:
            y = self.matmul(x)
            return y * self.scaling_factor + y
        if not self._prepared:
            self._prepare()
        M, K = x.shape
        N = self._w1t.shape[1]
        n_el = M * K
        x1 = torch.empty((M, K), device=x.device, dtype=torch.float16)
        x2 = torch.empty((M, K), device=x.device, dtype=torch.float16)
        _split_fp16_kernel[(triton.cdiv(n_el, 4096),)](
            x, x1, x2, n_el, BLOCK=4096, num_warps=8
        )
        out = torch.empty((M, N), device=x.device, dtype=torch.float32)
        even = (M % 128 == 0) and (N % 128 == 0) and (K % 32 == 0)
        grid = (triton.cdiv(M, 128) * triton.cdiv(N, 128),)
        _split_gemm_kernel[grid](
            x1, x2, self._w1t, self._w2t, self._bias, out,
            M, N, K,
            BM=128, BN=128, BK=32, GROUP=8, EVEN=even,
            num_warps=8, num_stages=5,
        )
        return out
