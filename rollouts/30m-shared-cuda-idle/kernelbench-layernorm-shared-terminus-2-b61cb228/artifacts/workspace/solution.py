import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _ln_p1(X, PS, NS_: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    row = pid // S
    part = pid % S
    base = row * NS_
    offs = part * BLOCK + tl.arange(0, BLOCK)
    xv = tl.load(X + base + offs)
    tl.store(PS + pid * 2 + 0, tl.sum(xv, axis=0))
    tl.store(PS + pid * 2 + 1, tl.sum(xv * xv, axis=0))


@triton.jit
def _ln_mid(PS, MR, NS_: tl.constexpr, S: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    poffs = row * S * 2 + tl.arange(0, S) * 2
    ps1 = tl.load(PS + poffs)
    ps2 = tl.load(PS + poffs + 1)
    total = tl.sum(ps1, axis=0)
    total2 = tl.sum(ps2, axis=0)
    mean = total / NS_
    var = total2 / NS_ - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    tl.store(MR + row * 2 + 0, mean)
    tl.store(MR + row * 2 + 1, rstd)


@triton.jit
def _ln_p2(X, W, Bv, Y, MR, NS_: tl.constexpr, BLOCK: tl.constexpr, RPC: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    row = pid // RPC
    rc = pid % RPC
    base = row * NS_
    mean = tl.load(MR + row * 2 + 0)
    rstd = tl.load(MR + row * 2 + 1)
    S: tl.constexpr = NS_ // BLOCK
    for part in range(rc, S, RPC):
        offs = part * BLOCK + tl.arange(0, BLOCK)
        xv = tl.load(X + base + offs)
        wv = tl.load(W + offs)
        bv = tl.load(Bv + offs)
        tl.store(Y + base + offs, (xv - mean) * rstd * wv + bv)


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.normalized_shape = tuple(normalized_shape)
        ns = 1
        for d in self.normalized_shape:
            ns *= d
        self.ns = ns
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        self.eps = 1e-5
        # pick split so each partial block is ~8192 elements
        s = max(1, ns // 8192)
        self.S = s
        self.ps = None
        self.mr = None
        self.wflat = None
        self.bflat = None

    def _ensure(self, x):
        rows = x.numel() // self.ns
        if self.ps is None or self.ps.shape[0] != rows * self.S * 2 or self.ps.device != x.device:
            self.ps = torch.empty(rows * self.S * 2, dtype=torch.float32, device=x.device)
            self.mr = torch.empty(rows * 2, dtype=torch.float32, device=x.device)
        if self.wflat is None or self.wflat.device != x.device:
            self.wflat = self.weight.detach().reshape(-1).contiguous()
            self.bflat = self.bias.detach().reshape(-1).contiguous()
        return rows

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        xc = x.contiguous()
        rows = self._ensure(xc)
        S = self.S
        NS_ = self.ns
        BLOCK = NS_ // S
        y = torch.empty_like(xc)
        _ln_p1[(rows * S,)](xc, self.ps, NS_=NS_, S=S, BLOCK=BLOCK, num_warps=16)
        _ln_mid[(rows,)](self.ps, self.mr, NS_=NS_, S=S, num_warps=4)
        RPC = 8
        B2 = 16384 if NS_ % 16384 == 0 else BLOCK
        _ln_p2[(rows * RPC,)](xc, self.wflat, self.bflat, y, self.mr, NS_=NS_, BLOCK=B2, RPC=RPC, num_warps=16)
        return y.view(orig_shape)
