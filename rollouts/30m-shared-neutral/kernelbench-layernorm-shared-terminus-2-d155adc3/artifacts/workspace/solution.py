import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _stats_kernel(x_ptr, ms_ptr, vs_ptr, D, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    i = tl.program_id(1)
    offs = i * BLOCK + tl.arange(0, BLOCK)
    mask = offs < D
    x = tl.load(x_ptr + row.to(tl.int64) * D + offs, mask=mask, other=0.0)
    s = tl.sum(x)
    s2 = tl.sum(x * x)
    tl.atomic_add(ms_ptr + row, s.to(tl.float64))
    tl.atomic_add(vs_ptr + row, s2.to(tl.float64))


@triton.jit
def _apply_kernel(x_ptr, out_ptr, w_ptr, b_ptr, ms_ptr, vs_ptr, D, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    i = tl.program_id(1)
    offs = i * BLOCK + tl.arange(0, BLOCK)
    mask = offs < D
    inv_d = 1.0 / D
    mean = (tl.load(ms_ptr + row) * inv_d).to(tl.float32)
    ex2 = (tl.load(vs_ptr + row) * inv_d).to(tl.float32)
    var = ex2 - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    x = tl.load(x_ptr + row.to(tl.int64) * D + offs, mask=mask)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    y = (x - mean) * rstd * w + b
    tl.store(out_ptr + row.to(tl.int64) * D + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.normalized_shape = tuple(normalized_shape)
        self.ln = nn.LayerNorm(normalized_shape=self.normalized_shape)
        self._graphs = {}
        self._bufs = {}
        self._capture_ok = True

    def _get_bufs(self, R, D, device):
        key = (R, D, str(device))
        if key not in self._bufs:
            self._bufs[key] = (
                torch.zeros(R, dtype=torch.float64, device=device),
                torch.zeros(R, dtype=torch.float64, device=device),
                torch.empty(R * D, dtype=torch.float32, device=device),
            )
        return self._bufs[key]

    def _launch(self, x2, out, w, b, ms, vs, R, D, S, eps, BLOCK):
        grid = (R, S)
        ms.zero_()
        vs.zero_()
        _stats_kernel[grid](x2, ms, vs, D, BLOCK=BLOCK, num_warps=16)
        _apply_kernel[grid](x2, out, w, b, ms, vs, D, eps, BLOCK=BLOCK, num_warps=8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            return self.ln(x)
        D = 1
        for s in self.normalized_shape:
            D *= s
        xc = x.contiguous()
        R = xc.numel() // D
        BLOCK = 8192
        S = triton.cdiv(D, BLOCK)
        eps = self.ln.eps
        w = self.ln.weight.view(-1)
        b = self.ln.bias.view(-1)
        ms, vs, outflat = self._get_bufs(R, D, xc.device)
        x2 = xc.view(R * D)
        out_shape = xc.shape
        key = (xc.data_ptr(), tuple(xc.shape), w.data_ptr(), b.data_ptr(), R, D)
        g = self._graphs.get(key)
        if g is None:
            self._launch(x2, outflat, w, b, ms, vs, R, D, S, eps, BLOCK)
            if self._capture_ok:
                torch.cuda.synchronize()
                try:
                    gr = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(gr):
                        self._launch(x2, outflat, w, b, ms, vs, R, D, S, eps, BLOCK)
                    if len(self._graphs) > 64:
                        self._graphs.clear()
                    self._graphs[key] = gr
                    gr.replay()
                except Exception:
                    self._capture_ok = False
        else:
            g.replay()
        return outflat.view(out_shape)
