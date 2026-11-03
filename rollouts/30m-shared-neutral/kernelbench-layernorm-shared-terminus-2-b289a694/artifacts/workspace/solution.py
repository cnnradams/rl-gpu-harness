import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


if _HAS_TRITON:
    @triton.jit
    def _ln_stats(X, PS, PSS, N, BPR: tl.constexpr,
                  BLOCK: tl.constexpr, ITERS: tl.constexpr):
        row = tl.program_id(0)
        blk = tl.program_id(1)
        base = blk * (BLOCK * ITERS)
        s = tl.zeros([BLOCK], dtype=tl.float32)
        ss = tl.zeros([BLOCK], dtype=tl.float32)
        for i in range(ITERS):
            offs = base + i * BLOCK + tl.arange(0, BLOCK)
            xx = tl.load(X + row * N + offs)
            s += xx
            ss += xx * xx
        tl.store(PS + row * BPR + blk, tl.sum(s))
        tl.store(PSS + row * BPR + blk, tl.sum(ss))

    @triton.jit
    def _ln_apply(X, W, B, Y, PS, PSS, N, eps, BPR: tl.constexpr,
                  BLOCK: tl.constexpr, ITERS: tl.constexpr):
        row = tl.program_id(0)
        blk = tl.program_id(1)
        poffs = tl.arange(0, BPR)
        s = tl.sum(tl.load(PS + row * BPR + poffs))
        ss = tl.sum(tl.load(PSS + row * BPR + poffs))
        mean = s / N
        var = ss / N - mean * mean
        rstd = 1.0 / tl.sqrt(var + eps)
        base = blk * (BLOCK * ITERS)
        for i in range(ITERS):
            offs = base + i * BLOCK + tl.arange(0, BLOCK)
            xx = tl.load(X + row * N + offs)
            ww = tl.load(W + offs)
            bb = tl.load(B + offs)
            tl.store(Y + row * N + offs, (xx - mean) * rstd * ww + bb)


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)
        self.normalized_shape = tuple(normalized_shape)
        n = 1
        for d in self.normalized_shape:
            n *= d
        self._n = n
        self._nd = len(self.normalized_shape)
        self._bufs = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not _HAS_TRITON) or (not x.is_cuda) or x.dtype != torch.float32 \
                or not x.is_contiguous() \
                or tuple(x.shape[-self._nd:]) != self.normalized_shape:
            return self.ln(x)
        N = self._n
        BLOCK = 8192
        ITERS = 4
        if N % (BLOCK * ITERS) != 0:
            return self.ln(x)
        rows = x.numel() // N
        BPR = N // (BLOCK * ITERS)
        key = (rows, x.device.index)
        buf = self._bufs.get(key)
        if buf is None:
            ps = torch.empty(rows * BPR, device=x.device, dtype=torch.float32)
            pss = torch.empty(rows * BPR, device=x.device, dtype=torch.float32)
            buf = (ps, pss)
            self._bufs[key] = buf
        ps, pss = buf
        y = torch.empty_like(x)
        w = self.ln.weight
        b = self.ln.bias
        grid = (rows, BPR)
        _ln_stats[grid](x, ps, pss, N, BPR=BPR, BLOCK=BLOCK, ITERS=ITERS,
                        num_warps=4, num_stages=2)
        _ln_apply[grid](x, w, b, y, ps, pss, N, self.ln.eps, BPR=BPR,
                        BLOCK=BLOCK, ITERS=ITERS, num_warps=4, num_stages=2)
        return y
