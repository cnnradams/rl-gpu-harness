import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _ln_stats_kernel(X, S1, S2, N, P, BLOCK: tl.constexpr, SPLIT: tl.constexpr):
    row = tl.program_id(0)
    part = tl.program_id(1)
    base = X + row.to(tl.int64) * N + part * (SPLIT * BLOCK)
    cols = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(SPLIT):
        x = tl.load(base + i * BLOCK + cols)
        acc += x
        acc2 += x * x
    tl.store(S1 + row * P + part, tl.sum(acc, axis=0))
    tl.store(S2 + row * P + part, tl.sum(acc2, axis=0))


@triton.jit
def _ln_apply_kernel(X, Y, W, B, S1, S2, N, P, eps,
                     BLOCK: tl.constexpr, SPLIT: tl.constexpr, PB: tl.constexpr):
    row = tl.program_id(0)
    part = tl.program_id(1)
    parts = tl.arange(0, PB)
    pmask = parts < P
    s1 = tl.load(S1 + row * P + parts, mask=pmask, other=0.0)
    s2 = tl.load(S2 + row * P + parts, mask=pmask, other=0.0)
    mean = tl.sum(s1, axis=0) / N
    var = tl.sum(s2, axis=0) / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    base = row.to(tl.int64) * N + part * (SPLIT * BLOCK)
    cols = tl.arange(0, BLOCK)
    wbase = part * (SPLIT * BLOCK) + cols
    for i in range(SPLIT):
        off = i * BLOCK
        x = tl.load(X + base + off + cols)
        w = tl.load(W + wbase + off)
        b = tl.load(B + wbase + off)
        tl.store(Y + base + off + cols, (x - mean) * rstd * w + b)


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.normalized_shape = tuple(normalized_shape)
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)
        n = 1
        for d in self.normalized_shape:
            n *= d
        self.N = n
        self.BLOCK = 2048
        self.SPLIT = 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.N
        per_part = self.BLOCK * self.SPLIT
        if (not x.is_cuda) or x.dtype != torch.float32 or n % per_part != 0:
            return self.ln(x)
        x = x.contiguous()
        batch = x.numel() // n
        P = n // per_part
        PB = triton.next_power_of_2(P)
        y = torch.empty_like(x)
        s1 = torch.empty(batch * P, device=x.device, dtype=torch.float32)
        s2 = torch.empty(batch * P, device=x.device, dtype=torch.float32)
        grid = (batch, P)
        _ln_stats_kernel[grid](x, s1, s2, n, P, BLOCK=self.BLOCK, SPLIT=self.SPLIT, num_warps=16)
        _ln_apply_kernel[grid](x, y, self.ln.weight, self.ln.bias, s1, s2, n, P,
                               self.ln.eps, BLOCK=self.BLOCK, SPLIT=self.SPLIT, PB=PB, num_warps=16)
        return y
