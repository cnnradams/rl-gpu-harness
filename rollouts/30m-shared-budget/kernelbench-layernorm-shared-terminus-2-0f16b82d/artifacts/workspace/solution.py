import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _ln_partial_kernel(x_ptr, part_ptr, N, S, BLOCK: tl.constexpr):
    row = tl.program_id(1)
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0)
    s = tl.sum(x, axis=0)
    q = tl.sum(x * x, axis=0)
    base = (row * S + pid) * 2
    tl.store(part_ptr + base, s)
    tl.store(part_ptr + base + 1, q)


@triton.jit
def _ln_apply_kernel(x_ptr, part_ptr, w_ptr, b_ptr, y_ptr, N, S, eps,
                     BLOCK: tl.constexpr, SP2: tl.constexpr, NCHUNK: tl.constexpr):
    row = tl.program_id(0)
    pid = tl.program_id(1)
    pids = tl.arange(0, SP2)
    pmask = pids < S
    ps = tl.load(part_ptr + (row * S + pids) * 2, mask=pmask, other=0.0)
    qs = tl.load(part_ptr + (row * S + pids) * 2 + 1, mask=pmask, other=0.0)
    mean = tl.sum(ps, axis=0) / N
    var = tl.sum(qs, axis=0) / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    start = pid * BLOCK * NCHUNK
    for i in tl.static_range(NCHUNK):
        offs = start + i * BLOCK + tl.arange(0, BLOCK)
        offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
        mask = offs < N
        x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0)
        y = (x - mean) * rstd * w + b
        tl.store(y_ptr + row * N + offs, y, mask=mask)


class ModelNew(nn.Module):
    """LayerNorm implemented with a split-row two-kernel Triton scheme."""

    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.normalized_shape = tuple(normalized_shape)
        N = 1
        for d in self.normalized_shape:
            N *= d
        self.N = N
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        self.eps = 1e-5
        self.B1 = 32768
        self.S = (N + self.B1 - 1) // self.B1
        self.SP2 = triton.next_power_of_2(self.S)
        self.B2 = 8192
        self.NC = 16
        chunk = self.B2 * self.NC
        self.S2 = (N + chunk - 1) // chunk
        self._parts = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = self.N
        B = x.numel() // N
        S = self.S
        device = x.device
        part = self._parts.get(device)
        if part is None or part.numel() < B * S * 2:
            part = torch.empty(B * S * 2, device=device, dtype=torch.float32)
            self._parts[device] = part
        y = torch.empty_like(x)
        _ln_partial_kernel[(S, B)](x, part, N, S, BLOCK=self.B1, num_warps=16)
        _ln_apply_kernel[(B, self.S2)](x, part, self.weight, self.bias, y, N, S,
                                       self.eps, BLOCK=self.B2, SP2=self.SP2,
                                       NCHUNK=self.NC, num_warps=4)
        return y
