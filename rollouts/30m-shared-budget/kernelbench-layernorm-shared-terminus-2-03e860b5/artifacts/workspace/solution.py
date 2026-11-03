import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _layernorm_fused(
    X, Y, W, B,
    eps,
    ROWS,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    x_base = X.to(tl.pointer_type(tl.float32))
    y_base = Y.to(tl.pointer_type(tl.float32))
    rows_per_sm = (ROWS + num_pids - 1) // num_pids
    row_start = pid * rows_per_sm
    row_end = tl.minimum(row_start + rows_per_sm, ROWS)

    for row in range(row_start, row_end):
        row_ptr = row.to(tl.int64) * N
        mean = tl.zeros((), dtype=tl.float32)
        m2 = tl.zeros((), dtype=tl.float32)
        cnt = 0
        for k in range(NUM_BLOCKS):
            offs = k * BLOCK_N + tl.arange(0, BLOCK_N)
            x = tl.load(x_base + row_ptr + offs)
            new_cnt = cnt + BLOCK_N
            delta = x - mean
            mean += tl.sum(delta) / new_cnt
            delta2 = x - mean
            m2 += tl.sum(delta * delta2)
            cnt = new_cnt
        rstd = 1.0 / tl.sqrt(m2 / N + eps)
        for k in range(NUM_BLOCKS):
            offs = k * BLOCK_N + tl.arange(0, BLOCK_N)
            x = tl.load(x_base + row_ptr + offs)
            w = tl.load(W + offs)
            b = tl.load(B + offs)
            y = (x - mean) * rstd * w + b
            tl.store(y_base + row_ptr + offs, y)


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)
        self.normalized_shape = tuple(normalized_shape)
        n = 1
        for s in self.normalized_shape:
            n *= s
        self.N = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        shape = x.shape
        nrows = x.numel() // self.N
        BLOCK_N = 16384
        if self.N % BLOCK_N != 0 or not x.is_cuda:
            return F.layer_norm(x, self.normalized_shape, self.ln.weight, self.ln.bias, self.ln.eps)
        x2 = x.view(nrows, self.N)
        y = torch.empty_like(x2)
        NUM_BLOCKS = self.N // BLOCK_N
        grid = (min(132, nrows),)
        _layernorm_fused[grid](
            x2, y, self.ln.weight, self.ln.bias,
            self.ln.eps,
            nrows,
            N=self.N,
            BLOCK_N=BLOCK_N,
            NUM_BLOCKS=NUM_BLOCKS,
            num_warps=16,
        )
        return y.view(shape)
