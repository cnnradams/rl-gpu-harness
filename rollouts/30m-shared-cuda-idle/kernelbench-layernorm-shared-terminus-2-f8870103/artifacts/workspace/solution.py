import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _ln_reduce_kernel(X, P, M, CHUNK, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    base = row * M + chunk * CHUNK
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, CHUNK, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        x = tl.load(X + base + idx)
        acc += x
        acc2 += x * x
    s = tl.sum(acc)
    s2 = tl.sum(acc2)
    S = tl.num_programs(1)
    tl.store(P + row * 2 * S + chunk, s)
    tl.store(P + row * 2 * S + S + chunk, s2)


@triton.jit
def _ln_norm_kernel(X, W, B, P, Y, M, CHUNK, eps,
                    S: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    offs = tl.arange(0, S)
    s = tl.sum(tl.load(P + row * 2 * S + offs))
    s2 = tl.sum(tl.load(P + row * 2 * S + S + offs))
    mean = s / M
    var = s2 / M - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    base_x = row * M + chunk * CHUNK
    base_w = chunk * CHUNK
    for off in range(0, CHUNK, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        x = tl.load(X + base_x + idx)
        w = tl.load(W + base_w + idx)
        b = tl.load(B + base_w + idx)
        y = (x - mean) * rstd * w + b
        tl.store(Y + base_x + idx, y)


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        self.normalized_shape = tuple(normalized_shape)
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        M = 1
        for d in self.normalized_shape:
            M *= d
        if (not x.is_cuda) or x.dtype != torch.float32 or (M % 16384) != 0:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.contiguous()
        rows = x.numel() // M
        S = M // 16384
        BLOCK = 8192
        w = self.weight.contiguous()
        b = self.bias.contiguous()
        y = torch.empty_like(x)
        P = torch.empty(rows * 2 * S, dtype=torch.float32, device=x.device)
        _ln_reduce_kernel[(rows, S)](x, P, M, M // S, BLOCK=BLOCK, num_warps=16)
        _ln_norm_kernel[(rows, S)](x, w, b, P, y, M, M // S, self.eps,
                                   S=S, BLOCK=BLOCK, num_warps=16)
        return y
