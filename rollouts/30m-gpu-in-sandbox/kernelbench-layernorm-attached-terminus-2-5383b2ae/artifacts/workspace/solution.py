import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _ln_stats(x_ptr, part_ptr,
              CHUNK: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * CHUNK
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(0, CHUNK, BLOCK):
        v = tl.load(x_ptr + base + i + tl.arange(0, BLOCK))
        acc += v
        acc2 += v * v
    tl.store(part_ptr + pid * 2 + 0, tl.sum(acc))
    tl.store(part_ptr + pid * 2 + 1, tl.sum(acc2))


@triton.jit
def _ln_norm(x_ptr, w_ptr, b_ptr, part_ptr, out_ptr,
             N: tl.constexpr, CHUNK: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr, EPS: tl.constexpr):
    pid = tl.num_programs(0) - 1 - tl.program_id(0)
    m = pid // S
    soffs = tl.arange(0, S)
    ps = tl.load(part_ptr + (m * S + soffs) * 2 + 0)
    ps2 = tl.load(part_ptr + (m * S + soffs) * 2 + 1)
    total = tl.sum(ps)
    total2 = tl.sum(ps2)
    mean = total / N
    var = total2 / N - mean * mean
    var = tl.maximum(var, 0.0) + EPS
    rstd = 1.0 / tl.sqrt(var)
    base = pid.to(tl.int64) * CHUNK
    row_base = m.to(tl.int64) * N
    for i in range(0, CHUNK, BLOCK):
        idx = i + tl.arange(0, BLOCK)
        v = tl.load(x_ptr + base + idx)
        w = tl.load(w_ptr + (base - row_base) + idx)
        b = tl.load(b_ptr + (base - row_base) + idx)
        y = (v - mean) * rstd * w + b
        tl.store(out_ptr + base + idx, y)


_S = 16
_BLOCK = 8192
_NUM_WARPS = 8


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.normalized_shape = tuple(normalized_shape)
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        self.eps = 1e-5
        self._N = 1
        for d in self.normalized_shape:
            self._N *= d
        self._state = {}
        self._part = None

    def _launch(self, x, out, part, M, N):
        S = _S
        BLOCK = _BLOCK
        CHUNK = N // S
        grid = (M * S,)
        _ln_stats[grid](x, part, CHUNK=CHUNK, BLOCK=BLOCK, num_warps=_NUM_WARPS)
        _ln_norm[grid](x, self.weight, self.bias, part, out,
                       N=N, CHUNK=CHUNK, S=S, BLOCK=BLOCK, EPS=self.eps, num_warps=_NUM_WARPS)

    def _get_part(self, M, device):
        S = _S
        if self._part is None or self._part.numel() < M * S * 2 or self._part.device != device:
            self._part = torch.empty(M * S * 2, dtype=torch.float32, device=device)
        return self._part

    def _forward_eager(self, x, M, N):
        out = torch.empty_like(x)
        self._launch(x, out, self._get_part(M, x.device), M, N)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = self._N
        if not (x.is_cuda and x.dtype == torch.float32 and x.is_contiguous()):
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        M = x.numel() // N
        if (N % (_S * _BLOCK)) != 0:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        key = (x.data_ptr(), tuple(x.shape), x.device.index)
        st = self._state.get(key)
        if st is None:
            self._state[key] = 1
            return self._forward_eager(x, M, N)
        if st == 'no_graph':
            return self._forward_eager(x, M, N)
        if st == 'graph':
            g, out = self._state[('g', key)]
            g.replay()
            return out
        # second call with this exact input buffer: capture a CUDA graph
        try:
            part = self._get_part(M, x.device)
            out = torch.empty_like(x)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._launch(x, out, part, M, N)
            g.replay()
            self._state[key] = 'graph'
            self._state[('g', key)] = (g, out)
            return out
        except Exception:
            self._state[key] = 'no_graph'
            return self._forward_eager(x, M, N)
