import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _split_f32(x_ptr, hi_ptr, lo_ptr, numel, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel
        v = tl.load(x_ptr + offs, mask=mask)
        hi = (v.to(tl.int32, bitcast=True) & -8192).to(tl.float32, bitcast=True)
        tl.store(hi_ptr + offs, hi, mask=mask)
        tl.store(lo_ptr + offs, v - hi, mask=mask)


def _veltkamp(v):
    hi = (v.view(torch.int32) & -8192).view(torch.float32)
    return hi, v - hi


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self._key = None
        self._w_hi_t = None
        self._w_lo_t = None
        self._bias_s = None
        self._bufs = {}

    def _refresh(self):
        w = self.matmul.weight
        b = self.matmul.bias
        key = (w.data_ptr(), b.data_ptr(), w.device, w.dtype, w.shape)
        if self._key != key:
            wt = w.detach().t().contiguous()  # (in, out)
            w_hi, w_lo = _veltkamp(wt)
            s = 1.0 + self.scaling_factor
            self._w_hi_t = w_hi.t().contiguous()  # (out, in)
            self._w_lo_t = w_lo.t().contiguous()
            self._bias_s = (b.detach() * s).contiguous()
            self._bufs = {}
            self._key = key

    def _get_bufs(self, M, K, N, device):
        key = (M, K, N, device)
        if key not in self._bufs:
            self._bufs[key] = (
                torch.empty(M, K, device=device, dtype=torch.float32),
                torch.empty(M, K, device=device, dtype=torch.float32),
                torch.empty(M, N, device=device, dtype=torch.float32),
            )
        return self._bufs[key]

    def forward(self, x):
        w = self.matmul.weight
        b = self.matmul.bias
        s = 1.0 + self.scaling_factor
        if (not _HAS_TRITON or not x.is_cuda or x.dtype != torch.float32
                or not x.is_contiguous() or x.dim() != 2):
            y = torch.nn.functional.linear(x, w, b)
            return y * self.scaling_factor + y

        self._refresh()
        M, K = x.shape
        N = w.shape[0]
        x_hi, x_lo, t = self._get_bufs(M, K, N, x.device)

        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            _split_f32[(triton.cdiv(M * K, 4096),)](
                x, x_hi, x_lo, M * K, BLOCK=4096, num_warps=8)
            torch.addmm(self._bias_s, x_hi, self._w_hi_t.t(), beta=1, alpha=s, out=t)
            torch.addmm(t, x_lo, self._w_hi_t.t(), beta=1, alpha=s, out=t)
            torch.addmm(t, x_hi, self._w_lo_t.t(), beta=1, alpha=s, out=t)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
        return t
