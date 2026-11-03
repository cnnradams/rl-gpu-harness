import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _ln_stats(x_ptr, part_ptr, N, R,
              B: tl.constexpr, BLOCK: tl.constexpr, ITERS: tl.constexpr):
    pid = tl.program_id(0)
    row = tl.program_id(1)
    base = row * N + pid * (BLOCK * ITERS)
    offs = tl.arange(0, BLOCK)
    acc1 = tl.zeros((BLOCK,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in tl.static_range(ITERS):
        x = tl.load(x_ptr + base + i * BLOCK + offs, eviction_policy='evict_first')
        acc1 += x
        acc2 += x * x
    s1 = tl.sum(acc1, axis=0)
    s2 = tl.sum(acc2, axis=0)
    tl.store(part_ptr + row * B + pid, s1)
    tl.store(part_ptr + R * B + row * B + pid, s2)


@triton.jit
def _ln_apply(x_ptr, y_ptr, w_ptr, b_ptr, part_ptr, N, R, eps,
              B: tl.constexpr, BLOCK: tl.constexpr):
    # grid = (R, N // BLOCK): consecutive blocks share the same w/b tile
    row = tl.program_id(0)
    pid = tl.program_id(1)
    bidx = tl.arange(0, B)
    s1 = tl.sum(tl.load(part_ptr + row * B + bidx), axis=0)
    s2 = tl.sum(tl.load(part_ptr + R * B + row * B + bidx), axis=0)
    mean = s1 / N
    var = s2 / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    offs = row * N + pid * BLOCK + tl.arange(0, BLOCK)
    woffs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, eviction_policy='evict_first')
    w = tl.load(w_ptr + woffs, eviction_policy='evict_last')
    bb = tl.load(b_ptr + woffs, eviction_policy='evict_last')
    y = (x - mean) * rstd * w + bb
    tl.store(y_ptr + offs, y, eviction_policy='evict_first')


class ModelNew(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.normalized_shape = tuple(normalized_shape)
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        self.eps = 1e-5
        N = 1
        for s in self.normalized_shape:
            N *= s
        self.N = N
        self.B = 128
        self.SBLOCK = 4096
        self.ITERS = max(1, N // (self.B * self.SBLOCK))
        self.ABLOCK = 4096
        self._ok = (self.B * self.SBLOCK * self.ITERS == N and N % self.ABLOCK == 0)
        self._part = None
        self._y = None
        self._R = -1
        self._graph = None
        self._graph_key = None

    def _launch(self, x):
        N = self.N
        R = self._R
        _ln_stats[(self.B, R)](x, self._part, N, R, B=self.B, BLOCK=self.SBLOCK,
                               ITERS=self.ITERS, num_warps=16)
        _ln_apply[(R, N // self.ABLOCK)](x, self._y, self.weight, self.bias,
                                         self._part, N, R, self.eps,
                                         B=self.B, BLOCK=self.ABLOCK, num_warps=1)

    def _prepare(self, x, R):
        if self._R != R or self._part is None or self._part.device != x.device:
            self._part = torch.empty((2, R, self.B), dtype=torch.float32, device=x.device)
            self._y = torch.empty_like(x)
            self._R = R
            self._graph = None
            self._graph_key = None

    def forward(self, x):
        if (not self._ok or x.dtype != torch.float32 or not x.is_contiguous()
                or x.requires_grad or not x.is_cuda):
            return torch.nn.functional.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps)
        R = x.numel() // self.N
        self._prepare(x, R)
        key = (x.data_ptr(), self.weight.data_ptr(), self.bias.data_ptr())
        if self._graph is None or self._graph_key != key:
            self._launch(x)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._launch(x)
            self._graph = g
            self._graph_key = key
        self._graph.replay()
        return self._y
