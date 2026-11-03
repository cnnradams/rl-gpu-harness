import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _stats_kernel(X, PSUM, PSQ, N,
                  CHUNK: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    base = row * N + chunk * CHUNK
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK], dtype=tl.float32)
    for off in tl.static_range(0, CHUNK, BLOCK):
        x = tl.load(X + base + off + tl.arange(0, BLOCK))
        acc += x
        acc2 += x * x
    nchunk = tl.num_programs(1)
    pid = row * nchunk + chunk
    tl.store(PSUM + pid, tl.sum(acc))
    tl.store(PSQ + pid, tl.sum(acc2))


@triton.jit
def _apply_kernel(X, W, B, Y, PSUM, PSQ, N, eps,
                  NCHUNK: tl.constexpr, NPOW: tl.constexpr,
                  CHUNK: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    offs = tl.arange(0, NPOW)
    mask = offs < NCHUNK
    ps = tl.load(PSUM + row * NCHUNK + offs, mask=mask, other=0.0)
    ps2 = tl.load(PSQ + row * NCHUNK + offs, mask=mask, other=0.0)
    mean = tl.sum(ps) / N
    var = tl.sum(ps2) / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    xbase = row * N + chunk * CHUNK
    wbase = chunk * CHUNK
    for off in tl.static_range(0, CHUNK, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        x = tl.load(X + xbase + idx)
        w = tl.load(W + wbase + idx)
        b = tl.load(B + wbase + idx)
        y = (x - mean) * rstd * w + b
        tl.store(Y + xbase + idx, y)


class ModelNew(nn.Module):
    _CHUNK = 16384
    _BLOCK = 4096

    def __init__(self, normalized_shape):
        super().__init__()
        self.normalized_shape = tuple(normalized_shape)
        self.ln = nn.LayerNorm(normalized_shape=self.normalized_shape)
        self._N = 1
        for s in self.normalized_shape:
            self._N *= s
        self._partials = {}
        self._graphs = {}

    def _run(self, x, N, rows):
        CHUNK = self._CHUNK
        BLOCK = self._BLOCK
        NCHUNK = N // CHUNK
        key = (rows, x.device.index)
        buf = self._partials.get(key)
        if buf is None:
            buf = (torch.empty(rows * NCHUNK, device=x.device, dtype=torch.float32),
                   torch.empty(rows * NCHUNK, device=x.device, dtype=torch.float32))
            self._partials[key] = buf
        psum, psq = buf
        y = torch.empty_like(x)
        grid = (rows, NCHUNK)
        _stats_kernel[grid](x, psum, psq, N,
                            CHUNK=CHUNK, BLOCK=BLOCK, num_warps=8)
        _apply_kernel[grid](x, self.ln.weight, self.ln.bias, y, psum, psq, N,
                            self.ln.eps, NCHUNK=NCHUNK,
                            NPOW=triton.next_power_of_2(NCHUNK),
                            CHUNK=CHUNK, BLOCK=BLOCK, num_warps=8)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda or not x.is_contiguous() or self._N % self._CHUNK != 0:
            return self.ln(x)
        N = self._N
        rows = x.numel() // N
        key = (x.data_ptr(), tuple(x.shape), x.device.index)
        entry = self._graphs.get(key)
        if entry is not None:
            g, out = entry
            g.replay()
            return out
        # compile kernels / warm up on a side stream
        s = torch.cuda.Stream(device=x.device)
        s.wait_stream(torch.cuda.current_stream(x.device))
        with torch.cuda.stream(s):
            for _ in range(2):
                self._run(x, N, rows)
        torch.cuda.current_stream(x.device).wait_stream(s)
        torch.cuda.synchronize(x.device)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = self._run(x, N, rows)
        g.replay()
        torch.cuda.synchronize(x.device)
        self._graphs[key] = (g, out)
        return out
