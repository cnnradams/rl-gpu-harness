import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _ln_stats_kernel(x_ptr, p_ptr, M, N, C,
                         BLOCK_N: tl.constexpr):
        pid = tl.program_id(0)
        row = pid // C
        cid = pid % C
        base = row.to(tl.int64) * N
        offs = cid * BLOCK_N + tl.arange(0, BLOCK_N)
        v = tl.load(x_ptr + base + offs)
        tl.store(p_ptr + row * C + cid, tl.sum(v))
        tl.store(p_ptr + M * C + row * C + cid, tl.sum(v * v))

    @triton.jit
    def _ln_reduce_kernel(p_ptr, mr_ptr, N, C, eps,
                          BLOCK_C: tl.constexpr):
        row = tl.program_id(0)
        M = tl.num_programs(0)
        coffs = tl.arange(0, BLOCK_C)
        cmask = coffs < C
        s1 = tl.sum(tl.load(p_ptr + row * C + coffs, mask=cmask, other=0.0))
        s2 = tl.sum(tl.load(p_ptr + M * C + row * C + coffs, mask=cmask, other=0.0))
        invN = 1.0 / N
        mean = s1 * invN
        var = s2 * invN - mean * mean
        rstd = 1.0 / tl.sqrt(var + eps)
        tl.store(mr_ptr + row * 2, mean)
        tl.store(mr_ptr + row * 2 + 1, rstd)

    @triton.jit
    def _ln_map_kernel(x_ptr, w_ptr, b_ptr, mr_ptr, y_ptr, M, N,
                       BLOCK_N: tl.constexpr):
        cid = tl.program_id(0)
        offs = cid * BLOCK_N + tl.arange(0, BLOCK_N)
        ww = tl.load(w_ptr + offs)
        bb = tl.load(b_ptr + offs)
        for r in range(M):
            base = r.to(tl.int64) * N
            mean = tl.load(mr_ptr + r * 2)
            rstd = tl.load(mr_ptr + r * 2 + 1)
            v = tl.load(x_ptr + base + offs)
            tl.store(y_ptr + base + offs, (v - mean) * rstd * ww + bb)


_STATS_BLOCK_N = 8192
_STATS_WARPS = 16
_MAP_BLOCK_N = 512
_MAP_WARPS = 8


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = 1
        for s in self.normalized_shape:
            N *= s
        if (_HAS_TRITON and x.is_cuda and x.dtype == torch.float32
                and x.is_contiguous() and x.numel() % N == 0 and N > 0
                and N % _STATS_BLOCK_N == 0 and N % _MAP_BLOCK_N == 0
                and (x.numel() // N) * N < 2**62):
            return self._triton_forward(x, N)
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

    def _triton_forward(self, x: torch.Tensor, N: int) -> torch.Tensor:
        M = x.numel() // N
        w = self.weight.contiguous().view(-1)
        b = self.bias.contiguous().view(-1)
        C = N // _STATS_BLOCK_N
        ws = torch.empty(2 * M * C + 2 * M, device=x.device, dtype=torch.float32)
        partials = ws[:2 * M * C]
        mr = ws[2 * M * C:]
        y = torch.empty_like(x)
        _ln_stats_kernel[(M * C,)](x, partials, M, N, C,
                                   BLOCK_N=_STATS_BLOCK_N,
                                   num_warps=_STATS_WARPS)
        _ln_reduce_kernel[(M,)](partials, mr, N, C, self.eps,
                                BLOCK_C=triton.next_power_of_2(C),
                                num_warps=4)
        _ln_map_kernel[(N // _MAP_BLOCK_N,)](x, w, b, mr, y, M, N,
                                             BLOCK_N=_MAP_BLOCK_N,
                                             num_warps=_MAP_WARPS)
        return y
