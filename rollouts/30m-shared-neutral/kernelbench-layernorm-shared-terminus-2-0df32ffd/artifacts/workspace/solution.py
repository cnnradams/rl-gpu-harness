import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _ln_stats_kernel(X, PART, N: tl.constexpr, NSPLIT: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // NSPLIT
    part = pid % NSPLIT
    CHUNK: tl.constexpr = N // NSPLIT
    base = X + row * N + part * CHUNK
    cols = tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_N], dtype=tl.float32)
    for i in range(0, CHUNK, BLOCK_N):
        x = tl.load(base + i + cols, eviction_policy='evict_first')
        acc += x
        acc2 += x * x
    tl.store(PART + pid * 2 + 0, tl.sum(acc, axis=0))
    tl.store(PART + pid * 2 + 1, tl.sum(acc2, axis=0))


@triton.jit
def _ln_apply_kernel(X, Y, W, B, PART, N: tl.constexpr, NSPLIT: tl.constexpr, BLOCK_N: tl.constexpr, EPS: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // NSPLIT
    part = pid % NSPLIT
    CHUNK: tl.constexpr = N // NSPLIT
    offs = tl.arange(0, NSPLIT)
    ps = tl.load(PART + (row * NSPLIT + offs) * 2 + 0)
    ps2 = tl.load(PART + (row * NSPLIT + offs) * 2 + 1)
    mean = tl.sum(ps, axis=0) / N
    var = tl.sum(ps2, axis=0) / N - mean * mean
    rstd = tl.rsqrt(var + EPS)
    x_base = X + row * N + part * CHUNK
    y_base = Y + row * N + part * CHUNK
    wb_base = part * CHUNK
    cols = tl.arange(0, BLOCK_N)
    for i in range(0, CHUNK, BLOCK_N):
        x = tl.load(x_base + i + cols, eviction_policy='evict_first')
        w = tl.load(W + wb_base + i + cols)
        b = tl.load(B + wb_base + i + cols)
        y = (x - mean) * rstd * w + b
        tl.store(y_base + i + cols, y, eviction_policy='evict_first')


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)
        self.normalized_shape = tuple(normalized_shape)
        N = 1
        for s in self.normalized_shape:
            N *= s
        self.N = N
        # choose config
        cfg = None
        if N % 16 == 0:
            for nsplit in (16, 8, 4, 2, 1):
                if N % nsplit != 0:
                    continue
                chunk = N // nsplit
                for bn in (4096, 2048, 1024, 512, 256, 128, 64, 32, 16):
                    if bn <= chunk and chunk % bn == 0:
                        cfg = (nsplit, bn)
                        break
                if cfg:
                    break
        self.cfg = cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda or self.cfg is None:
            return self.ln(x)
        x = x.contiguous()
        N = self.N
        M = x.numel() // N
        NSPLIT, BLOCK_N = self.cfg
        w = self.ln.weight
        b = self.ln.bias
        if w.dtype != torch.float32 or not w.is_contiguous():
            w = w.float().contiguous()
            b = b.float().contiguous()
        y = torch.empty_like(x)
        part = torch.empty(M * NSPLIT * 2, device=x.device, dtype=torch.float32)
        _ln_stats_kernel[(M * NSPLIT,)](x, part, N, NSPLIT, BLOCK_N, num_warps=8)
        _ln_apply_kernel[(M * NSPLIT,)](x, y, w, b, part, N, NSPLIT, BLOCK_N, 1e-5, num_warps=8)
        return y
