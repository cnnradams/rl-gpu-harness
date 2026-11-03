import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _ln_stats_kernel(
    X, W, Bs, PS, PQ,
    C: tl.constexpr,
    INNER: tl.constexpr,
    BLOCK: tl.constexpr,
    VEC: tl.constexpr,
):
    pid = tl.program_id(0)
    ch = pid % C
    N: tl.constexpr = BLOCK * VEC
    offs = tl.arange(0, N)
    x_ptr = X + pid.to(tl.int64) * INNER
    w_ptr = W + ch * INNER
    b_ptr = Bs + ch * INNER
    s = tl.zeros((N,), dtype=tl.float64)
    sq = tl.zeros((N,), dtype=tl.float64)
    for k in range(0, INNER // N):
        o = k * N + offs
        xx = tl.load(x_ptr + o).to(tl.float64)
        ww = tl.load(w_ptr + o).to(tl.float64)
        bb = tl.load(b_ptr + o).to(tl.float64)
        yy = xx * ww + bb
        s += yy
        sq += yy * yy
    tl.store(PS + pid, tl.sum(s, axis=0))
    tl.store(PQ + pid, tl.sum(sq, axis=0))


@triton.jit
def _ln_reduce_kernel(
    PS, PQ, Mean, Rstd, eps,
    C: tl.constexpr,
    S: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float64)
    acc2 = tl.zeros((BLOCK,), dtype=tl.float64)
    for k in range(0, C // BLOCK):
        o = k * BLOCK + offs
        acc += tl.load(PS + b * C + o)
        acc2 += tl.load(PQ + b * C + o)
    total = tl.sum(acc, axis=0)
    total2 = tl.sum(acc2, axis=0)
    mean = total / S
    var = total2 / S - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(Mean + b, mean.to(tl.float32))
    tl.store(Rstd + b, rstd.to(tl.float32))


@triton.jit
def _ln_apply_kernel(
    X, W, Bs, Mean, Rstd, Y,
    S: tl.constexpr,
    INNER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    base = pid * BLOCK
    b = base // S
    j = base % INNER
    offs = base + tl.arange(0, BLOCK)
    offsj = j + tl.arange(0, BLOCK)
    xx = tl.load(X + offs)
    ww = tl.load(W + offsj)
    bb = tl.load(Bs + offsj)
    m = tl.load(Mean + b)
    r = tl.load(Rstd + b)
    yy = (xx * ww + bb - m) * r
    tl.store(Y + offs, yy)


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        B = shape[0]
        S = 1
        for d in shape[1:]:
            S *= d
        C = self.ln.weight.shape[0]
        INNER = S // C
        if S % C != 0 or INNER % 1024 != 0 or S % 4096 != 0 or C % 16 != 0:
            return self.ln(x)
        x2 = x.reshape(B * C, INNER)
        w = self.ln.weight
        bias = self.ln.bias
        ps = torch.empty(B * C, device=x.device, dtype=torch.float64)
        pq = torch.empty(B * C, device=x.device, dtype=torch.float64)
        mean = torch.empty(B, device=x.device, dtype=torch.float32)
        rstd = torch.empty(B, device=x.device, dtype=torch.float32)
        y = torch.empty(B * C, INNER, device=x.device, dtype=x.dtype)
        _ln_stats_kernel[(B * C,)](
            x2, w, bias, ps, pq,
            C=C, INNER=INNER, BLOCK=512, VEC=2, num_warps=2,
        )
        _ln_reduce_kernel[(B,)](
            ps, pq, mean, rstd, self.ln.eps,
            C=C, S=S, BLOCK=16, num_warps=1,
        )
        _ln_apply_kernel[(B * S // 4096,)](
            x2, w, bias, mean, rstd, y,
            S=S, INNER=INNER, BLOCK=4096, num_warps=16,
        )
        return y.reshape(shape)
