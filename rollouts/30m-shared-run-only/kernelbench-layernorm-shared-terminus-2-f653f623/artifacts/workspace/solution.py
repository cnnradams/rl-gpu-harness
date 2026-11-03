import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _ln_stats_kernel(
    x_ptr, part_mean_ptr, part_m2_ptr, count_ptr,
    numel_per_row, NCHUNKS: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // NCHUNKS
    chunk = pid % NCHUNKS
    row_start = row * numel_per_row
    chunk_size = (numel_per_row + NCHUNKS - 1) // NCHUNKS
    start = chunk * chunk_size
    end = tl.minimum(start + chunk_size, numel_per_row)
    m = tl.zeros((), dtype=tl.float32)
    m2 = tl.zeros((), dtype=tl.float32)
    n = tl.zeros((), dtype=tl.float32)
    for off in range(start, end, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        v = tl.load(x_ptr + row_start + idx)
        n_new = n + BLOCK
        delta = v - m
        m += tl.sum(delta, axis=0) / n_new
        m2 += tl.sum(delta * (v - m), axis=0)
        n = n_new
    tl.store(part_mean_ptr + pid, m)
    tl.store(part_m2_ptr + pid, m2)
    tl.store(count_ptr + pid, n)


@triton.jit
def _ln_reduce_kernel(
    part_mean_ptr, part_m2_ptr, count_ptr, mean_ptr, rstd_ptr,
    numel_per_row, eps, NCHUNKS: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, NCHUNKS)
    base = row * NCHUNKS
    m = tl.load(part_mean_ptr + base + offs)
    m2 = tl.load(part_m2_ptr + base + offs)
    n = tl.load(count_ptr + base + offs)
    total = tl.sum(n, axis=0)
    mean = tl.sum(m * n, axis=0) / total
    delta = m - mean
    M2 = tl.sum(m2 + n * delta * delta, axis=0)
    rstd = 1.0 / tl.sqrt(M2 / total + eps)
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def _ln_apply_kernel(
    x_ptr, mean_ptr, rstd_ptr, w_ptr, b_ptr, y_ptr,
    C: tl.constexpr, BLOCK: tl.constexpr, N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    wv = tl.load(w_ptr + offs)
    bv = tl.load(b_ptr + offs)
    for r in range(C):
        base = r * N + offs
        v = tl.load(x_ptr + base)
        mean = tl.load(mean_ptr + r)
        rstd = tl.load(rstd_ptr + r)
        tl.store(y_ptr + base, (v - mean) * rstd * wv + bv)


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.normalized_shape = tuple(normalized_shape)
        self.ln = nn.LayerNorm(normalized_shape=self.normalized_shape)
        self._part = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        numel_per_row = 1
        for s in self.ln.normalized_shape:
            numel_per_row *= s
        total = x.numel()
        C = total // numel_per_row
        w = self.ln.weight
        b = self.ln.bias
        eps = self.ln.eps
        x2 = x.view(-1)
        y = torch.empty_like(x)
        y2 = y.view(-1)
        NCHUNKS = 32
        part = self._part
        if part is None or part.shape[0] < C * NCHUNKS * 3:
            part = torch.empty(C * NCHUNKS * 3, device=x.device, dtype=torch.float32)
            self._part = part
        part_mean = part[:C * NCHUNKS]
        part_m2 = part[C * NCHUNKS:2 * C * NCHUNKS]
        count = part[2 * C * NCHUNKS:3 * C * NCHUNKS]
        mr = torch.empty(2 * C, device=x.device, dtype=torch.float32)
        mean = mr[:C]
        rstd = mr[C:]
        BLOCK1 = 8192
        _ln_stats_kernel[(C * NCHUNKS,)](
            x2, part_mean, part_m2, count, numel_per_row,
            NCHUNKS=NCHUNKS, BLOCK=BLOCK1, num_warps=8, num_stages=5,
        )
        _ln_reduce_kernel[(C,)](
            part_mean, part_m2, count, mean, rstd, numel_per_row, eps,
            NCHUNKS=NCHUNKS, num_warps=1,
        )
        BLOCK2 = 8192
        if numel_per_row % BLOCK2 == 0 and C <= 128:
            _ln_apply_kernel[(numel_per_row // BLOCK2,)](
                x2, mean, rstd, w, b, y2, C=C, BLOCK=BLOCK2, N=numel_per_row,
                num_warps=16, num_stages=2,
            )
        else:
            y = torch.nn.functional.layer_norm(x, self.ln.normalized_shape, w, b, eps)
        return y
