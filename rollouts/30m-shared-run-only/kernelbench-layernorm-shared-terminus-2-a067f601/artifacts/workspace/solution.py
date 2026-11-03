import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _stats_kernel(X, PS, PS2, CHUNK: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // S
    ch = pid % S
    base = row * (CHUNK * S) + ch * CHUNK
    acc1 = tl.zeros((BLOCK,), tl.float32)
    acc2 = tl.zeros((BLOCK,), tl.float32)
    for off in tl.static_range(0, CHUNK, BLOCK):
        v = tl.load(X + base + off + tl.arange(0, BLOCK))
        acc1 += v
        acc2 += v * v
    tl.store(PS + pid, tl.sum(acc1))
    tl.store(PS2 + pid, tl.sum(acc2))


@triton.jit
def _norm_kernel(X, Y, W, Bias, PS, PS2, eps,
                 CHUNK: tl.constexpr, S: tl.constexpr, SPOW: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // S
    ch = pid % S
    offs = tl.arange(0, SPOW)
    pm = offs < S
    p1 = tl.load(PS + row * S + offs, mask=pm, other=0.0)
    p2 = tl.load(PS2 + row * S + offs, mask=pm, other=0.0)
    NN = CHUNK * S
    mean = tl.sum(p1) / NN
    var = tl.sum(p2) / NN - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    base = row * NN + ch * CHUNK
    wbase = ch * CHUNK
    for off in tl.static_range(0, CHUNK, BLOCK):
        c = off + tl.arange(0, BLOCK)
        v = tl.load(X + base + c)
        w = tl.load(W + wbase + c)
        b = tl.load(Bias + wbase + c)
        tl.store(Y + base + c, (v - mean) * rstd * w + b)


class ModelNew(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        self.eps = 1e-5
        n = 1
        for d in self.normalized_shape:
            n *= d
        self.N = n
        self.chunk = 131072
        self.block = 8192
        self.warps = 4
        self.stages = 3
        self._bufs = {}
        self._graph = None
        self._gkey = None
        self._pending = None
        self._gout = None

    def _run(self, x):
        rows = x.numel() // self.N
        S = self.N // self.chunk
        bkey = (x.device.index, rows)
        bufs = self._bufs.get(bkey)
        if bufs is None:
            bufs = (torch.empty(rows * S, device=x.device, dtype=torch.float32),
                    torch.empty(rows * S, device=x.device, dtype=torch.float32))
            self._bufs[bkey] = bufs
        ps, ps2 = bufs
        y = torch.empty_like(x)
        w = self.weight.reshape(-1)
        b = self.bias.reshape(-1)
        _stats_kernel[(rows * S,)](x, ps, ps2, CHUNK=self.chunk, S=S, BLOCK=self.block,
                                   num_warps=self.warps, num_stages=self.stages)
        _norm_kernel[(rows * S,)](x, y, w, b, ps, ps2, self.eps,
                                  CHUNK=self.chunk, S=S, SPOW=triton.next_power_of_2(S),
                                  BLOCK=self.block, num_warps=self.warps, num_stages=self.stages)
        return y

    def forward(self, x):
        rows = x.numel() // self.N
        if (not x.is_cuda) or x.dtype != torch.float32 or (self.N % self.chunk) != 0 or rows * self.N != x.numel():
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.contiguous()
        key = (x.data_ptr(), tuple(x.shape), self.weight.data_ptr(), self.bias.data_ptr(), x.device.index)
        if self._graph is not None and key == self._gkey:
            self._graph.replay()
            return self._gout
        if key == self._pending:
            try:
                self._run(x)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self._gout = self._run(x)
                self._graph = g
                self._gkey = key
                g.replay()
                return self._gout
            except Exception:
                self._pending = None
                self._graph = None
                self._gkey = None
                return self._run(x)
        self._pending = key
        return self._run(x)
