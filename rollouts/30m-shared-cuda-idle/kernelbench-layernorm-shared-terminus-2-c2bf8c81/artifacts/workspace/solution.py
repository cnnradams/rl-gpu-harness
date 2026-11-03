import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _ln_split_kernel(X, Y, W, B, FSUM, CNT, N, eps,
                     S: tl.constexpr, CHUNK: tl.constexpr,
                     AFF: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // S
    sid = pid % S
    offs = sid * CHUNK + tl.arange(0, CHUNK)
    x = tl.load(X + row * N + offs)
    s = tl.sum(x, axis=0)
    sq = tl.sum(x * x, axis=0)
    tl.atomic_add(FSUM + 2 * row, s, sem="release")
    tl.atomic_add(FSUM + 2 * row + 1, sq, sem="release")
    c = tl.atomic_add(CNT + row, 1, sem="release")
    while c < S:
        c = tl.atomic_add(CNT + row, 0, sem="acquire")
    total = tl.atomic_add(FSUM + 2 * row, 0.0, sem="acquire")
    totsq = tl.atomic_add(FSUM + 2 * row + 1, 0.0, sem="acquire")
    mean = total / N
    var = totsq / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    if AFF:
        w = tl.load(W + offs)
        b = tl.load(B + offs)
        y = (x - mean) * rstd * w + b
    else:
        y = (x - mean) * rstd
    tl.store(Y + row * N + offs, y)


class ModelNew(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.normalized_shape = tuple(normalized_shape)
        self.ln = nn.LayerNorm(normalized_shape=self.normalized_shape)
        self._ident = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.ln.weight
        b = self.ln.bias
        eps = self.ln.eps
        N = w.numel()
        if (not x.is_cuda) or x.dtype != torch.float32:
            return F.layer_norm(x, self.normalized_shape, w, b, eps)
        S = 128
        if N % S != 0:
            return F.layer_norm(x, self.normalized_shape, w, b, eps)
        if self._ident is None:
            self._ident = bool((w.detach() == 1).all().item()) and bool((b.detach() == 0).all().item())
        x = x.contiguous()
        M = x.numel() // N
        CHUNK = N // S
        y = torch.empty_like(x)
        fsum = torch.zeros(2 * M, device=x.device, dtype=torch.float32)
        cnt = torch.zeros(M, device=x.device, dtype=torch.int32)
        _ln_split_kernel[(M * S,)](x, y, w, b, fsum, cnt, N, eps,
                                   S=S, CHUNK=CHUNK, AFF=self._ident,
                                   num_warps=32)
        return y
