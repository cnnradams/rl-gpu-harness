import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False

_BLOCK = 8192
_W1 = 16
_W2 = 16
_WR = 8
_ROWS = 16
_MAX_GRAPHS = 32

if _HAS_TRITON:
    @triton.jit
    def _ln_part(X, WS, N, NB, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        b = pid // NB
        o = (pid % NB) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + b * N + o)
        tl.store(WS + pid * 2, tl.sum(x))
        tl.store(WS + pid * 2 + 1, tl.sum(x * x))

    @triton.jit
    def _ln_reduce(WS, MR, NB, eps, N, NBV: tl.constexpr):
        b = tl.program_id(0)
        ar = tl.arange(0, NBV)
        s = tl.sum(tl.load(WS + b * NB * 2 + ar * 2))
        sq = tl.sum(tl.load(WS + b * NB * 2 + ar * 2 + 1))
        inv = 1.0 / N
        mean = s * inv
        var = sq * inv - mean * mean
        tl.store(MR + b * 2, mean)
        tl.store(MR + b * 2 + 1, 1.0 / tl.sqrt(var + eps))

    @triton.jit
    def _ln_norm(X, Y, W, B, MR, N, NB, R: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        b = pid % R
        c = pid // R
        mean = tl.load(MR + b * 2)
        rstd = tl.load(MR + b * 2 + 1)
        o = c * BLOCK + tl.arange(0, BLOCK)
        offs = b * N + o
        x = tl.load(X + offs)
        w = tl.load(W + o)
        bb = tl.load(B + o)
        tl.store(Y + offs, (x - mean) * rstd * w + bb)


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)
        self.normalized_shape = tuple(normalized_shape)
        self._fn = None
        self._graphs = {}
        if _HAS_TRITON:
            try:
                n = 1
                for s in self.normalized_shape:
                    n *= s
                if len(self.normalized_shape) == 3 and n % _BLOCK == 0:
                    nb = n // _BLOCK

                    def fn(x, w, b, ws, mr, y, _R=_ROWS, _NB=nb, _N=n, _eps=self.ln.eps):
                        _ln_part[(_R * _NB,)](x, ws, _N, _NB, BLOCK=_BLOCK, num_warps=_W1)
                        _ln_reduce[(_R,)](ws, mr, _NB, _eps, _N, NBV=_NB, num_warps=_WR)
                        _ln_norm[(_R * _NB,)](x, y, w, b, mr, _N, _NB, R=_R, BLOCK=_BLOCK, num_warps=_W2)

                    self._fn = fn
            except Exception:
                self._fn = None

    def _plain(self, x):
        nb = (x.numel() // x.shape[0]) // _BLOCK
        ws = torch.empty(x.shape[0] * nb * 2, device=x.device, dtype=torch.float32)
        mr = torch.empty(x.shape[0] * 2, device=x.device, dtype=torch.float32)
        y = torch.empty_like(x)
        self._fn(x, self.ln.weight, self.ln.bias, ws, mr, y)
        return y

    def _make_graph(self, x):
        nb = (x.numel() // x.shape[0]) // _BLOCK
        ws = torch.empty(x.shape[0] * nb * 2, device=x.device, dtype=torch.float32)
        mr = torch.empty(x.shape[0] * 2, device=x.device, dtype=torch.float32)
        y = torch.empty_like(x)
        # eager warm run (also produces this call's result in y)
        self._fn(x, self.ln.weight, self.ln.bias, ws, mr, y)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            gr.capture_begin()
            self._fn(x, self.ln.weight, self.ln.bias, ws, mr, y)
            gr.capture_end()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        return (gr, y, ws, mr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = self.normalized_shape
        if not (
            _HAS_TRITON
            and self._fn is not None
            and x.is_cuda
            and x.dtype == torch.float32
            and x.dim() == 4
            and x.shape[1:] == shape
            and x.is_contiguous()
            and x.shape[0] == _ROWS
            and self.ln.weight is not None
            and self.ln.bias is not None
            and self.ln.weight.is_contiguous()
            and self.ln.bias.is_contiguous()
        ):
            return F.layer_norm(x, shape, self.ln.weight, self.ln.bias, self.ln.eps)
        try:
            capturing = torch.cuda.is_current_stream_capturing()
        except Exception:
            capturing = True
        if capturing:
            return self._plain(x)
        key = (x.data_ptr(), tuple(x.shape))
        ent = self._graphs.get(key)
        if ent is None:
            if len(self._graphs) >= _MAX_GRAPHS:
                return self._plain(x)
            try:
                ent = self._make_graph(x)
            except Exception:
                return self._plain(x)
            self._graphs[key] = ent
            return ent[1]
        ent[0].replay()
        return ent[1]
