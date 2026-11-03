import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _stats_kernel(x_ptr, psum_ptr, psq_ptr, N,
                  ITERS: tl.constexpr, CB: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)
    ncols = tl.num_programs(1)
    base = row.to(tl.int64) * N + col.to(tl.int64) * (ITERS * CB)
    acc = tl.zeros([CB], tl.float32)
    acc2 = tl.zeros([CB], tl.float32)
    for i in tl.static_range(ITERS):
        off = base + i * CB + tl.arange(0, CB)
        v = tl.load(x_ptr + off)
        acc += v
        acc2 += v * v
    tl.store(psum_ptr + row * ncols + col, tl.sum(acc))
    tl.store(psq_ptr + row * ncols + col, tl.sum(acc2))


@triton.jit
def _apply_kernel(x_ptr, w_ptr, b_ptr, o_ptr, psum_ptr, psq_ptr, N, eps,
                  C: tl.constexpr, ITERS: tl.constexpr, CB: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)
    idx = tl.arange(0, C)
    s = tl.sum(tl.load(psum_ptr + row * C + idx))
    s2 = tl.sum(tl.load(psq_ptr + row * C + idx))
    mean = s / N
    var = s2 / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    base = row.to(tl.int64) * N + col.to(tl.int64) * (ITERS * CB)
    wbase = col.to(tl.int64) * (ITERS * CB)
    for i in tl.static_range(ITERS):
        off = i * CB + tl.arange(0, CB)
        v = tl.load(x_ptr + base + off)
        wv = tl.load(w_ptr + wbase + off)
        bv = tl.load(b_ptr + wbase + off)
        y = (v - mean) * (rstd * wv) + bv
        tl.store(o_ptr + base + off, y)


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.normalized_shape = tuple(normalized_shape)
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)
        self.eps = self.ln.eps

    def _forward_triton(self, x):
        N = 1
        for d in self.normalized_shape:
            N *= d
        R = x.numel() // N
        CB1, IT1 = 2048, 64
        CB2, IT2 = 8192, 2
        C = N // (CB1 * IT1)
        C2 = N // (CB2 * IT2)
        if (C < 1 or C > 1024 or N % (CB1 * IT1) != 0
                or C2 < 1 or N % (CB2 * IT2) != 0):
            return self.ln(x)
        out = torch.empty_like(x)
        psum = torch.empty(R, C, device=x.device, dtype=torch.float32)
        psq = torch.empty(R, C, device=x.device, dtype=torch.float32)
        xf = x.view(-1)
        wf = self.ln.weight.view(-1)
        bf = self.ln.bias.view(-1)
        of = out.view(-1)
        _stats_kernel[(R, C)](xf, psum, psq, N, ITERS=IT1, CB=CB1, num_warps=4)
        _apply_kernel[(R, C2)](xf, wf, bf, of, psum, psq, N, self.eps,
                               C=C, ITERS=IT2, CB=CB2, num_warps=4)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and x.dtype == torch.float32 and x.is_contiguous():
            return self._forward_triton(x)
        return self.ln(x)
