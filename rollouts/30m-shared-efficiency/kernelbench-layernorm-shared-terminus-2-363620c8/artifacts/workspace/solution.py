import os
import torch
import torch.nn as nn
import triton
import triton.language as tl

P_ENV = int(os.environ.get('LN_P', '512'))


@triton.jit
def _fused_ln(X, Y, W, B, WS, BAR, target,
              ROWS: tl.constexpr, N: tl.constexpr, eps,
              P: tl.constexpr, CHUNK: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * CHUNK
    for r in range(ROWS):
        Xb = X + r * N
        acc = tl.zeros([BLOCK], tl.float32)
        acc2 = tl.zeros([BLOCK], tl.float32)
        for off in range(start, start + CHUNK, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            v = tl.load(Xb + cols)
            acc += v
            acc2 += v * v
        tl.store(WS + (r * P + pid) * 2, tl.sum(acc))
        tl.store(WS + (r * P + pid) * 2 + 1, tl.sum(acc2))
        # barrier: release arrive, poll with volatile L2 load, acquire at end
        tl.atomic_add(BAR + r, 1, sem='release', scope='gpu')
        while tl.load(BAR + r, volatile=True) < target:
            pass
        tl.atomic_add(BAR + r, 0, sem='acquire', scope='gpu')
        parts = tl.arange(0, P)
        s = tl.load(WS + (r * P + parts) * 2)
        s2 = tl.load(WS + (r * P + parts) * 2 + 1)
        mean = tl.sum(s) / N
        var = tl.sum(s2) / N - mean * mean
        rstd = 1.0 / tl.sqrt(var + eps)
        Yb = Y + r * N
        for off in range(start, start + CHUNK, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            v = tl.load(Xb + cols)
            wv = tl.load(W + cols, cache_modifier='.ca')
            bv = tl.load(B + cols, cache_modifier='.ca')
            tl.store(Yb + cols, (v - mean) * rstd * wv + bv, cache_modifier='.cs')


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        self.normalized_shape = tuple(normalized_shape)
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)
        self._ws = None
        self._bar = None
        self._call = 0
        self._P = P_ENV

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = 1
        for s in self.normalized_shape:
            N *= s
        rows = x.numel() // N
        P = self._P
        CHUNK = (N + P - 1) // P
        BLOCK = 2048
        if (N % CHUNK == 0) and x.is_contiguous() and x.is_cuda and (P & (P - 1)) == 0:
            y = torch.empty_like(x)
            if self._ws is None or self._ws.shape[0] != rows * P * 2 or self._ws.device != x.device:
                self._ws = torch.empty(rows * P * 2, dtype=torch.float32, device=x.device)
                self._bar = torch.zeros(rows, dtype=torch.int32, device=x.device)
                self._call = 0
            self._call += 1
            target = self._call * P
            _fused_ln[(P,)](x, y, self.ln.weight, self.ln.bias, self._ws, self._bar, target,
                            ROWS=rows, N=N, eps=self.ln.eps,
                            P=P, CHUNK=CHUNK, BLOCK=BLOCK, num_warps=8)
            return y
        return nn.functional.layer_norm(x, self.normalized_shape, self.ln.weight, self.ln.bias, self.ln.eps)
