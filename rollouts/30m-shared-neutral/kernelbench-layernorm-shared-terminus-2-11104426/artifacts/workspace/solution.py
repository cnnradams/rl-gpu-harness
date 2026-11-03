import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _ln_stats(x_ptr, s_ptr, N, R,
              BLOCK: tl.constexpr, C: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // C
    c = pid % C
    base = row * N + c * (N // C)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N // C, BLOCK):
        v = tl.load(x_ptr + base + off + tl.arange(0, BLOCK))
        acc += v
        acc2 += v * v
    tl.store(s_ptr + row * C + c, tl.sum(acc, axis=0))
    tl.store(s_ptr + R * C + row * C + c, tl.sum(acc2, axis=0))


@triton.jit
def _ln_apply(x_ptr, w_ptr, b_ptr, s_ptr, y_ptr, N, R, eps,
              BLOCK: tl.constexpr, C: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // C
    c = pid % C
    idx = tl.arange(0, C)
    s1 = tl.load(s_ptr + row * C + idx)
    s2 = tl.load(s_ptr + R * C + row * C + idx)
    mean = tl.sum(s1, axis=0) / N
    var = tl.sum(s2, axis=0) / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    base = row * N + c * (N // C)
    for off in range(0, N // C, BLOCK):
        i = base + off + tl.arange(0, BLOCK)
        v = tl.load(x_ptr + i)
        w = tl.load(w_ptr + i - row * N)
        b = tl.load(b_ptr + i - row * N)
        tl.store(y_ptr + i, (v - mean) * rstd * w + b)


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)
        N = 1
        for s in tuple(normalized_shape):
            N *= s
        self._N = N
        self._cfg = None
        for C, BLOCK in ((16, 8192), (8, 8192), (4, 8192), (2, 8192), (1, 8192),
                         (1, 4096), (1, 2048), (1, 1024)):
            if N % (C * BLOCK) == 0:
                self._cfg = (C, BLOCK)
                break

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._cfg is None or not x.is_cuda or x.dtype != torch.float32:
            return self.ln(x)
        N = self._N
        C, BLOCK = self._cfg
        x2 = x.contiguous().view(-1, N)
        R = x2.shape[0]
        y = torch.empty_like(x2)
        stats = torch.empty(2 * R * C, device=x.device, dtype=torch.float32)
        w = self.ln.weight
        b = self.ln.bias
        _ln_stats[(R * C,)](x2, stats, N, R, BLOCK=BLOCK, C=C,
                            num_warps=4, num_stages=4)
        _ln_apply[(R * C,)](x2, w, b, stats, y, N, R, self.ln.eps,
                            BLOCK=BLOCK, C=C, num_warps=4, num_stages=4)
        return y.view_as(x)
