import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _ln_partial_kernel(x_ptr, part_ptr, R, C: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // C
    c = pid % C
    offs = c * BLOCK + tl.arange(0, BLOCK)
    base = row.to(tl.int64) * R
    x = tl.load(x_ptr + base + offs)
    s1 = tl.sum(x)
    s2 = tl.sum(x * x)
    tl.store(part_ptr + pid * 2 + 0, s1)
    tl.store(part_ptr + pid * 2 + 1, s2)


@triton.jit
def _ln_apply_kernel(x_ptr, y_ptr, part_ptr, w_ptr, b_ptr, R,
                     eps, invR,
                     C: tl.constexpr, BLOCK: tl.constexpr, NW: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // C
    c = pid % C
    base = row * 2 * C + tl.arange(0, C) * 2
    s1v = tl.load(part_ptr + base)
    s2v = tl.load(part_ptr + base + 1)
    mean = tl.sum(s1v) * invR
    var = tl.sum(s2v) * invR - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    offs = c * BLOCK + tl.arange(0, BLOCK)
    xb = row.to(tl.int64) * R
    x = tl.load(x_ptr + xb + offs)
    widx = c * BLOCK + tl.arange(0, BLOCK)
    if NW != 0:
        w = tl.load(w_ptr + widx)
        b = tl.load(b_ptr + widx)
        y = (x - mean) * rstd * w + b
    else:
        y = (x - mean) * rstd
    tl.store(y_ptr + xb + offs, y)


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.ln = nn.LayerNorm(normalized_shape=self.normalized_shape)
        self.eps = self.ln.eps
        self._C = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        R = 1
        for s in self.normalized_shape:
            R *= s
        N = x.numel() // R
        C = self._C
        BLOCK = R // C
        parts = torch.empty(N * C * 2, device=x.device, dtype=torch.float32)
        _ln_partial_kernel[(N * C,)](x, parts, R, C=C, BLOCK=BLOCK, num_warps=8)
        y = torch.empty_like(x)
        w = self.ln.weight
        b = self.ln.bias
        nw = 1 if (w is not None and w.numel() == R) else 0
        _ln_apply_kernel[(N * C,)](x, y, parts,
                                   w if nw else x, b if nw else x,
                                   R, float(self.eps), 1.0 / R,
                                   C=C, BLOCK=BLOCK, NW=nw, num_warps=8)
        return y
