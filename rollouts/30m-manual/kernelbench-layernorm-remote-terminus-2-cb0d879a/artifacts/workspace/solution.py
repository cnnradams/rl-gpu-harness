import os
import torch
import torch.nn as nn
import triton
import triton.language as tl

_NC = int(os.environ.get('LN_NUM_CHUNKS', '16'))
_NW = int(os.environ.get('LN_NUM_WARPS', '16'))


@triton.jit
def _ln_stats_kernel(x_ptr, partials_ptr, N, num_chunks, CHUNK, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // num_chunks
    chunk = pid % num_chunks
    start = row * N + chunk * CHUNK
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in range(0, CHUNK, BLOCK):
        v = tl.load(x_ptr + start + i + offs)
        acc += v
        acc2 += v * v
    s = tl.sum(acc, axis=0)
    s2 = tl.sum(acc2, axis=0)
    tl.store(partials_ptr + row * num_chunks * 2 + chunk * 2 + 0, s)
    tl.store(partials_ptr + row * num_chunks * 2 + chunk * 2 + 1, s2)


@triton.jit
def _ln_norm_kernel(x_ptr, w_ptr, b_ptr, y_ptr, partials_ptr, N, num_chunks, eps,
                    CHUNK, BLOCK: tl.constexpr, NP2: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // num_chunks
    chunk = pid % num_chunks
    poffs = tl.arange(0, NP2)
    pmask = poffs < num_chunks
    s = tl.load(partials_ptr + row * num_chunks * 2 + poffs * 2 + 0, mask=pmask, other=0.0)
    s2 = tl.load(partials_ptr + row * num_chunks * 2 + poffs * 2 + 1, mask=pmask, other=0.0)
    mean = tl.sum(s, axis=0) / N
    mean2 = tl.sum(s2, axis=0) / N
    var = mean2 - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    start = row * N + chunk * CHUNK
    offs = tl.arange(0, BLOCK)
    for i in range(0, CHUNK, BLOCK):
        idx = start + i + offs
        v = tl.load(x_ptr + idx)
        w = tl.load(w_ptr + chunk * CHUNK + i + offs)
        b = tl.load(b_ptr + chunk * CHUNK + i + offs)
        y = (v - mean) * rstd * w + b
        tl.store(y_ptr + idx, y)


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)
        self._partials = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        N = 1
        for d in self.ln.normalized_shape:
            N *= d
        B = x.numel() // N
        num_chunks = _NC
        CHUNK = N // num_chunks
        BLOCK = 8192
        while CHUNK % BLOCK != 0 and BLOCK > 256:
            BLOCK //= 2
        w = self.ln.weight.reshape(-1)
        b = self.ln.bias.reshape(-1)
        y = torch.empty_like(x)
        if self._partials is None or self._partials.shape[0] < B * num_chunks * 2 or self._partials.device != x.device:
            self._partials = torch.empty(B * num_chunks * 2, dtype=torch.float32, device=x.device)
        partials = self._partials
        grid = (B * num_chunks,)
        _ln_stats_kernel[grid](x, partials, N, num_chunks, CHUNK, BLOCK, num_warps=_NW)
        _ln_norm_kernel[grid](x, w, b, y, partials, N, num_chunks, self.ln.eps, CHUNK, BLOCK, NP2=triton.next_power_of_2(num_chunks), num_warps=_NW)
        return y
