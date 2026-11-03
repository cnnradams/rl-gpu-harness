import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _stats_kernel(x_ptr, s_ptr, q_ptr, N,
                  BLOCK: tl.constexpr, NLOOP: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(0)
    row = tl.program_id(1)
    sacc = tl.zeros([BLOCK], dtype=tl.float32)
    qacc = tl.zeros([BLOCK], dtype=tl.float32)
    base_row = row.to(tl.int64) * N
    for i in tl.static_range(NLOOP):
        offs = (pid * NLOOP + i) * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            v = tl.load(x_ptr + base_row + offs, eviction_policy='evict_first')
        else:
            v = tl.load(x_ptr + base_row + offs, mask=offs < N, other=0.0, eviction_policy='evict_first')
        sacc += v
        qacc += v * v
    tl.atomic_add(s_ptr + row, tl.sum(sacc, axis=0))
    tl.atomic_add(q_ptr + row, tl.sum(qacc, axis=0))


@triton.jit
def _apply_kernel(x_ptr, w_ptr, b_ptr, s_ptr, q_ptr, o_ptr, N, eps,
                  BLOCK: tl.constexpr, NLOOP: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(0)
    row = tl.program_id(1)
    s = tl.load(s_ptr + row)
    q = tl.load(q_ptr + row)
    inv_n = 1.0 / N
    mean = s * inv_n
    var = q * inv_n - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    base_row = row.to(tl.int64) * N
    for i in tl.static_range(NLOOP):
        offs = (pid * NLOOP + i) * BLOCK + tl.arange(0, BLOCK)
        base = base_row + offs
        if EVEN:
            xv = tl.load(x_ptr + base, eviction_policy='evict_first')
            wv = tl.load(w_ptr + offs)
            bv = tl.load(b_ptr + offs)
            tl.store(o_ptr + base, (xv - mean) * rstd * wv + bv, eviction_policy='evict_first')
        else:
            mask = offs < N
            xv = tl.load(x_ptr + base, mask=mask, other=0.0, eviction_policy='evict_first')
            wv = tl.load(w_ptr + offs, mask=mask, other=0.0)
            bv = tl.load(b_ptr + offs, mask=mask, other=0.0)
            tl.store(o_ptr + base, (xv - mean) * rstd * wv + bv, mask=mask, eviction_policy='evict_first')


class ModelNew(nn.Module):
    def __init__(self, normalized_shape):
        super(ModelNew, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)
        self.eps = self.ln.eps
        self._acc = None
        self._key = None

    def _get_acc(self, rows, device):
        key = (rows, device)
        if self._key != key:
            self._acc = torch.zeros(2, rows, device=device, dtype=torch.float32)
            self._key = key
        return self._acc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.ln.weight
        b = self.ln.bias
        if not x.is_cuda or x.dtype != torch.float32:
            return torch.nn.functional.layer_norm(x, w.shape, w, b, self.eps)
        N = w.numel()
        if not x.is_contiguous():
            x = x.contiguous()
        rows = x.numel() // N
        out = torch.empty_like(x)
        acc = self._get_acc(rows, x.device)
        acc.zero_()
        wf = w.reshape(-1)
        bf = b.reshape(-1)
        SB, SW, SL = 4096, 8, 8
        AB, AW, AL = 4096, 8, 16
        even_s = (N % (SB * SL)) == 0
        even_a = (N % (AB * AL)) == 0
        _stats_kernel[(triton.cdiv(N, SB * SL), rows)](x, acc[0], acc[1], N,
                                                       BLOCK=SB, NLOOP=SL, EVEN=even_s, num_warps=SW)
        _apply_kernel[(triton.cdiv(N, AB * AL), rows)](x, wf, bf, acc[0], acc[1], out, N, self.eps,
                                                       BLOCK=AB, NLOOP=AL, EVEN=even_a, num_warps=AW)
        return out
