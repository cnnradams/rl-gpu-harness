import torch
import torch.nn as nn
import triton
import triton.language as tl

EPS = tl.constexpr(1e-5)

@triton.jit
def ln_partial(x_ptr, ws_ptr, ws2_ptr, N: tl.constexpr, SEG: tl.constexpr, SPS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // SPS
    seg = pid % SPS
    base = row.to(tl.int64) * N + seg * SEG
    s = tl.zeros([BLOCK], tl.float32)
    s2 = tl.zeros([BLOCK], tl.float32)
    for off in range(0, SEG, BLOCK):
        v = tl.load(x_ptr + base + off + tl.arange(0, BLOCK))
        s += v
        s2 += v * v
    tl.store(ws_ptr + pid, tl.sum(s))
    tl.store(ws2_ptr + pid, tl.sum(s2))

@triton.jit
def ln_apply_fused(x_ptr, y_ptr, w_ptr, b_ptr, ws_ptr, ws2_ptr, N: tl.constexpr, SEG: tl.constexpr, SPS: tl.constexpr, BLOCK: tl.constexpr, BS: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // SPS
    seg = pid % SPS
    col0 = seg * SEG
    base = row.to(tl.int64) * N + col0
    o = row * SPS + tl.arange(0, BS)
    msk = tl.arange(0, BS) < SPS
    s = tl.sum(tl.load(ws_ptr + o, mask=msk, other=0.0))
    s2 = tl.sum(tl.load(ws2_ptr + o, mask=msk, other=0.0))
    mean = s / N
    var = s2 / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + EPS)
    for off in range(0, SEG, BLOCK):
        oo = off + tl.arange(0, BLOCK)
        v = tl.load(x_ptr + base + oo)
        ww = tl.load(w_ptr + col0 + oo)
        bb = tl.load(b_ptr + col0 + oo)
        tl.store(y_ptr + base + oo, (v - mean) * rstd * ww + bb)


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        self.normalized_shape = tuple(normalized_shape)
        self.ln = nn.LayerNorm(normalized_shape=self.normalized_shape)
        self._N = 1
        for d in self.normalized_shape:
            self._N *= d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        N = self._N
        lead = x.numel() // N
        xf = x.view(lead, N)
        w = self.ln.weight
        b = self.ln.bias
        SEG = 262144
        if N % SEG != 0:
            return self.ln(x)
        SPS = N // SEG
        nblk = lead * SPS
        BLOCK = 2048
        dev = x.device
        ws = torch.empty(nblk, device=dev, dtype=torch.float32)
        ws2 = torch.empty(nblk, device=dev, dtype=torch.float32)
        out = torch.empty_like(xf)
        ln_partial[(nblk,)](xf, ws, ws2, N=N, SEG=SEG, SPS=SPS, BLOCK=BLOCK, num_warps=16)
        BS = 1
        while BS < SPS:
            BS *= 2
        ln_apply_fused[(nblk,)](xf, out, w, b, ws, ws2, N=N, SEG=SEG, SPS=SPS, BLOCK=BLOCK, BS=BS, num_warps=16)
        return out.view_as(x)
