import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _ln_partial(X, PS, PSQ, N: tl.constexpr, NSPLIT, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = tl.program_id(1)
    base = row.to(tl.int64) * N
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + base + offs)
    tl.store(PS + row * NSPLIT + pid, tl.sum(x))
    tl.store(PSQ + row * NSPLIT + pid, tl.sum(x * x))


@triton.jit
def _ln_apply(X, PS, PSQ, W, B, Y, N: tl.constexpr, M: tl.constexpr, NSPLIT, eps, BLOCK: tl.constexpr, NPOW: tl.constexpr):
    pid = tl.program_id(0)
    chunk = pid // M
    row = pid % M
    base = row.to(tl.int64) * N
    offs2 = tl.arange(0, NPOW)
    mask2 = offs2 < NSPLIT
    ps = tl.load(PS + row * NSPLIT + offs2, mask=mask2, other=0.0)
    psq = tl.load(PSQ + row * NSPLIT + offs2, mask=mask2, other=0.0)
    mean = tl.sum(ps) / N
    var = tl.sum(psq) / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    offs = chunk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + base + offs)
    w = tl.load(W + offs)
    b = tl.load(B + offs)
    tl.store(Y + base + offs, (x - mean) * rstd * w + b)


_BLOCK = 4096
_WARPS = 8


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = self.ln.eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = 1
        for s in self.normalized_shape:
            N *= s
        if (not x.is_cuda) or x.dtype != torch.float32 or (N % _BLOCK) != 0 or not x.is_contiguous():
            return self.ln(x)
        M = x.numel() // N
        x2 = x.view(M, N)
        y2 = torch.empty_like(x2)
        w = self.ln.weight.reshape(-1)
        b = self.ln.bias.reshape(-1)
        NSPLIT = N // _BLOCK
        ps = torch.empty((M, NSPLIT), device=x.device, dtype=torch.float32)
        psq = torch.empty((M, NSPLIT), device=x.device, dtype=torch.float32)
        _ln_partial[(NSPLIT, M)](x2, ps, psq, N, NSPLIT, BLOCK=_BLOCK, num_warps=_WARPS)
        _ln_apply[(NSPLIT * M,)](x2, ps, psq, w, b, y2, N, M, NSPLIT, self.eps,
                                 BLOCK=_BLOCK, NPOW=triton.next_power_of_2(NSPLIT), num_warps=_WARPS)
        return y2.view(x.shape)
