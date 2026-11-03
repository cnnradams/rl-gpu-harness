import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _ln_partials(X, S, N: tl.constexpr, COUNT: tl.constexpr, BLOCK: tl.constexpr, SPLIT: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // SPLIT
    part = pid % SPLIT
    base = row * N + part * COUNT
    acc = tl.zeros((BLOCK,), tl.float32)
    acc2 = tl.zeros((BLOCK,), tl.float32)
    for off in range(0, COUNT, BLOCK):
        x = tl.load(X + base + off + tl.arange(0, BLOCK))
        acc += x
        acc2 += x * x
    tl.store(S + row * SPLIT * 2 + part * 2 + 0, tl.sum(acc))
    tl.store(S + row * SPLIT * 2 + part * 2 + 1, tl.sum(acc2))


@triton.jit
def _ln_normalize(X, Y, W, B, S, eps, N: tl.constexpr, COUNT: tl.constexpr, BLOCK: tl.constexpr, SPLIT: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // SPLIT
    part = pid % SPLIT
    idx = tl.arange(0, SPLIT)
    s = tl.load(S + row * SPLIT * 2 + idx * 2)
    s2 = tl.load(S + row * SPLIT * 2 + idx * 2 + 1)
    mean = tl.sum(s) / N
    var = tl.sum(s2) / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    base = row * N + part * COUNT
    for off in range(0, COUNT, BLOCK):
        i = base + off + tl.arange(0, BLOCK)
        j = part * COUNT + off + tl.arange(0, BLOCK)
        x = tl.load(X + i)
        w = tl.load(W + j)
        b = tl.load(B + j)
        tl.store(Y + i, (x - mean) * rstd * w + b)


class ModelNew(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)
        self._n = 1
        for d in self.ln.normalized_shape:
            self._n *= d
        self._split = 16
        self._block = 4096

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = self._n
        SPLIT = self._split
        BLOCK = self._block
        if (not x.is_contiguous()) or x.dtype != torch.float32 or (N % (SPLIT * BLOCK)) != 0:
            return self.ln(x)
        shape = x.shape
        rows = x.numel() // N
        xf = x.reshape(-1)
        y = torch.empty_like(xf)
        S = torch.empty(rows * SPLIT * 2, device=x.device, dtype=torch.float32)
        COUNT = N // SPLIT
        w = self.ln.weight.reshape(-1)
        b = self.ln.bias.reshape(-1)
        grid = (rows * SPLIT,)
        _ln_partials[grid](xf, S, N, COUNT, BLOCK, SPLIT, num_warps=8, num_stages=4)
        _ln_normalize[grid](xf, y, w, b, S, self.ln.eps, N, COUNT, BLOCK, SPLIT, num_warps=8, num_stages=4)
        return y.view(shape)
