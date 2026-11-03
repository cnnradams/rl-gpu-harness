import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _stats_kernel(x_ptr, psum_ptr, psumsq_ptr, N,
                  NROW: tl.constexpr, CHUNK: tl.constexpr, SUB: tl.constexpr):
    pid = tl.program_id(0)
    rows = tl.arange(0, NROW)
    cols0 = pid * CHUNK + tl.arange(0, SUB)
    acc = tl.zeros((NROW, SUB), dtype=tl.float32)
    acc2 = tl.zeros((NROW, SUB), dtype=tl.float32)
    for i in tl.static_range(CHUNK // SUB):
        cols = cols0 + i * SUB
        t = tl.load(x_ptr + rows[:, None] * N + cols[None, :],
                    eviction_policy="evict_first")
        acc += t
        acc2 += t * t
    s = tl.sum(acc, axis=1)
    s2 = tl.sum(acc2, axis=1)
    tl.store(psum_ptr + pid * NROW + rows, s)
    tl.store(psumsq_ptr + pid * NROW + rows, s2)


@triton.jit
def _finalize_kernel(psum_ptr, psumsq_ptr, mean_ptr, rstd_ptr, eps,
                     N, NROW: tl.constexpr, NPART: tl.constexpr):
    r = tl.program_id(0)
    idx = tl.arange(0, NPART)
    s = tl.load(psum_ptr + idx * NROW + r).to(tl.float64)
    s2 = tl.load(psumsq_ptr + idx * NROW + r).to(tl.float64)
    total = tl.sum(s)
    total2 = tl.sum(s2)
    mean = total / N
    var = total2 / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(mean_ptr + r, mean.to(tl.float32))
    tl.store(rstd_ptr + r, rstd.to(tl.float32))


@triton.jit
def _norm_kernel(x_ptr, out_ptr, mean_ptr, rstd_ptr, w_ptr, b_ptr, N,
                 NROW: tl.constexpr, CHUNK: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * CHUNK + tl.arange(0, CHUNK)
    wv = tl.load(w_ptr + cols)
    bv = tl.load(b_ptr + cols)
    for r in tl.static_range(NROW):
        offs = r * N + cols
        m = tl.load(mean_ptr + r)
        rs = tl.load(rstd_ptr + r)
        xv = tl.load(x_ptr + offs)
        tl.store(out_ptr + offs, (xv - m) * rs * wv + bv)


_NPART = 256
_SUB = 2048
_NORM_CHUNK = 4096


class ModelNew(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.normalized_shape = tuple(normalized_shape)
        self.ln = nn.LayerNorm(self.normalized_shape)
        self._N = 1
        for s in self.normalized_shape:
            self._N *= s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = self._N
        M = x.numel() // N
        w = self.ln.weight
        b = self.ln.bias
        # fast path: fp32 contiguous CUDA tensor, 16 rows, divisible sizes
        if (x.is_cuda and x.dtype == torch.float32 and x.is_contiguous()
                and M == 16 and N % _NORM_CHUNK == 0
                and N % _NPART == 0 and (N // _NPART) % _SUB == 0):
            x2 = x.view(M, N)
            psum = torch.empty(_NPART * M, device=x.device, dtype=torch.float32)
            psum2 = torch.empty(_NPART * M, device=x.device, dtype=torch.float32)
            mean = torch.empty(M, device=x.device, dtype=torch.float32)
            rstd = torch.empty(M, device=x.device, dtype=torch.float32)
            out = torch.empty_like(x2)
            wf = w.reshape(-1)
            bf = b.reshape(-1)
            _stats_kernel[(_NPART,)](
                x2, psum, psum2, N,
                NROW=M, CHUNK=N // _NPART, SUB=_SUB, num_warps=8)
            _finalize_kernel[(M,)](
                psum, psum2, mean, rstd, self.ln.eps, N,
                NROW=M, NPART=_NPART, num_warps=4)
            _norm_kernel[(N // _NORM_CHUNK,)](
                x2, out, mean, rstd, wf, bf, N,
                NROW=M, CHUNK=_NORM_CHUNK, num_warps=8)
            return out.view(x.shape)
        return F.layer_norm(x, self.normalized_shape, w, b, self.ln.eps)
