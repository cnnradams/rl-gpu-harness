import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _reduce_kernel(x_ptr, part_ptr, N, CHUNK: tl.constexpr, BLOCK: tl.constexpr, S: tl.constexpr):
    row = tl.program_id(0)
    sp = tl.program_id(1)
    base = row * N + sp * CHUNK
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK,), dtype=tl.float32)
    for off in range(0, CHUNK, BLOCK):
        v = tl.load(x_ptr + base + off + tl.arange(0, BLOCK))
        acc += v
        acc2 += v * v
    s = tl.sum(acc)
    s2 = tl.sum(acc2)
    tl.store(part_ptr + (row * S + sp) * 2, s)
    tl.store(part_ptr + (row * S + sp) * 2 + 1, s2)


@triton.jit
def _norm_kernel(x_ptr, w_ptr, b_ptr, part_ptr, out_ptr, N, eps,
                 CHUNK: tl.constexpr, BLOCK: tl.constexpr, S: tl.constexpr):
    row = tl.program_id(0)
    sp = tl.program_id(1)
    offs = tl.arange(0, S)
    ps = tl.load(part_ptr + (row * S + offs) * 2)
    ps2 = tl.load(part_ptr + (row * S + offs) * 2 + 1)
    mean = tl.sum(ps) / N
    var = tl.sum(ps2) / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    base = row * N + sp * CHUNK
    col = sp * CHUNK
    for off in range(0, CHUNK, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        v = tl.load(x_ptr + base + idx)
        w = tl.load(w_ptr + col + idx)
        b = tl.load(b_ptr + col + idx)
        tl.store(out_ptr + base + idx, (v - mean) * rstd * w + b)


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.normalized_shape = tuple(normalized_shape)
        n = 1
        for d in self.normalized_shape:
            n *= d
        self.N = n
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        self.eps = 1e-5
        self._S = 256
        self._BLOCK = 4096
        self._warps = 16
        self._part = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        R = x.numel() // self.N
        S = self._S
        if self.N % S != 0:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        CHUNK = self.N // S
        BLOCK = self._BLOCK
        while BLOCK > CHUNK:
            BLOCK //= 2
        if CHUNK % BLOCK != 0:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        dev = x.device
        if self._part is None or self._part.device != dev or self._part.shape[0] != R * S * 2:
            self._part = torch.empty(R * S * 2, device=dev, dtype=torch.float32)
        part = self._part
        out = torch.empty_like(x)
        x2 = x.reshape(-1)
        o2 = out.reshape(-1)
        w = self.weight.reshape(-1)
        b = self.bias.reshape(-1)
        _reduce_kernel[(R, S)](x2, part, self.N, CHUNK=CHUNK, BLOCK=BLOCK, S=S, num_warps=self._warps)
        _norm_kernel[(R, S)](x2, w, b, part, o2, self.N, self.eps,
                             CHUNK=CHUNK, BLOCK=BLOCK, S=S, num_warps=self._warps)
        return out
