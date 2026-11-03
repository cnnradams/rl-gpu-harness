import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _ln_stats(x_ptr, ps_ptr, psq_ptr, N, chunk, C,
              BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // C
    ch = pid % C
    start = row * N + ch * chunk
    s = tl.zeros((), dtype=tl.float32)
    sq = tl.zeros((), dtype=tl.float32)
    for off in range(0, chunk, BLOCK):
        idx = start + off + tl.arange(0, BLOCK)
        v = tl.load(x_ptr + idx)
        s += tl.sum(v)
        sq += tl.sum(v * v)
    tl.store(ps_ptr + pid, s)
    tl.store(psq_ptr + pid, sq)


@triton.jit
def _ln_apply(x_ptr, w_ptr, b_ptr, y_ptr, ps_ptr, psq_ptr,
              N, chunk, C, eps,
              BLOCK: tl.constexpr, CB: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // C
    ch = pid % C
    co = tl.arange(0, CB)
    cmask = co < C
    s = tl.sum(tl.load(ps_ptr + row * C + co, mask=cmask, other=0.0))
    sq = tl.sum(tl.load(psq_ptr + row * C + co, mask=cmask, other=0.0))
    mean = s / N
    var = sq / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    wrow = ch * chunk
    xrow = row * N + ch * chunk
    for off in range(0, chunk, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        v = tl.load(x_ptr + xrow + idx)
        w = tl.load(w_ptr + wrow + idx)
        b = tl.load(b_ptr + wrow + idx)
        y = (v - mean) * rstd * w + b
        tl.store(y_ptr + xrow + idx, y)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)
        self._C = 16
        self._BLOCK = 16384
        self._warps_stats = 16
        self._warps_apply = 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if (not x.is_cuda) or x.dtype != torch.float32:
                return self.ln(x)
            x = x.contiguous()
            N = self.ln.weight.numel()
            rows = x.numel() // N
            C = self._C
            if N % C != 0 or (N // C) % self._BLOCK != 0:
                return self.ln(x)
            chunk = N // C
            dev = x.device
            y = torch.empty_like(x)
            ps = torch.empty(rows * C, device=dev, dtype=torch.float32)
            psq = torch.empty(rows * C, device=dev, dtype=torch.float32)
            w = self.ln.weight.detach().reshape(-1)
            b = self.ln.bias.detach().reshape(-1)
            grid = (rows * C,)
            _ln_stats[grid](x, ps, psq, N, chunk, C,
                            BLOCK=self._BLOCK, num_warps=self._warps_stats)
            _ln_apply[grid](x, w, b, y, ps, psq, N, chunk, C, self.ln.eps,
                            BLOCK=self._BLOCK, CB=_next_pow2(C),
                            num_warps=self._warps_apply)
            return y
