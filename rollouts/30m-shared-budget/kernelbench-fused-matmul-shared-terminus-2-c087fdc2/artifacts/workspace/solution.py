import torch
import torch.nn as nn
import torch.nn.functional as F

_kernel_ready = [False]

def _ensure_kernel():
    if _kernel_ready[0]:
        return True
    try:
        import triton
        import triton.language as tl

        @triton.autotune(
            configs=[
                triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "GROUP_M": 8}, num_warps=8, num_stages=3),
                triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_M": 8}, num_warps=8, num_stages=3),
                triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_M": 8}, num_warps=8, num_stages=4),
                triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_M": 8}, num_warps=4, num_stages=4),
            ],
            key=["M", "N", "K"],
        )
        @triton.jit
        def _gemm_fp16x3(X, WH, WL, BIAS, Y, M, N, K,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                         GROUP_M: tl.constexpr, EVEN_K: tl.constexpr):
            pid = tl.program_id(0)
            grid_m = tl.cdiv(M, BLOCK_M)
            grid_n = tl.cdiv(N, BLOCK_N)
            width = GROUP_M * grid_n
            group_id = pid // width
            group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
            pid_m = group_id * GROUP_M + (pid % group_size)
            pid_n = (pid % width) // group_size
            rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            rk = tl.arange(0, BLOCK_K)
            m_mask = rm < M
            n_mask = rn < N
            x_ptrs = X + rm[:, None] * K + rk[None, :]
            ah_ptrs = WH + rn[None, :] * K + rk[:, None]
            al_ptrs = WL + rn[None, :] * K + rk[:, None]
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BLOCK_K)):
                if EVEN_K:
                    x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
                    ah = tl.load(ah_ptrs, mask=n_mask[None, :], other=0.0)
                    al = tl.load(al_ptrs, mask=n_mask[None, :], other=0.0)
                else:
                    k_mask = rk < K - k * BLOCK_K
                    x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
                    ah = tl.load(ah_ptrs, mask=n_mask[None, :] & k_mask[:, None], other=0.0)
                    al = tl.load(al_ptrs, mask=n_mask[None, :] & k_mask[:, None], other=0.0)
                xh = x.to(tl.float16)
                xl = (x - xh.to(tl.float32)).to(tl.float16)
                acc = tl.dot(xh, ah, acc)
                acc = tl.dot(xh, al, acc)
                acc = tl.dot(xl, ah, acc)
                x_ptrs += BLOCK_K
                ah_ptrs += BLOCK_K
                al_ptrs += BLOCK_K
            bias = tl.load(BIAS + rn, mask=n_mask, other=0.0)
            acc = acc + bias[None, :]
            y_ptrs = Y + rm[:, None] * N + rn[None, :]
            tl.store(y_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])

        def _split_gemm(x, wh, wl, bias):
            M, K = x.shape
            N = wh.shape[0]
            y = torch.empty((M, N), device=x.device, dtype=torch.float32)
            grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
            _gemm_fp16x3[grid](x, wh, wl, bias, y, M, N, K, BLOCK_K=64, EVEN_K=(K % 64 == 0))
            return y

        globals()["_split_gemm"] = _split_gemm
        _kernel_ready[0] = True
        return True
    except Exception:
        return False


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self._wh = None
        self._wl = None
        self._bias = None
        self._K = -1

    @torch.no_grad()
    def _prepare(self):
        w = self.matmul.weight
        b = self.matmul.bias
        if w.device.type != "cuda" or w.dtype != torch.float32:
            return
        factor = 1.0 + float(self.scaling_factor)
        w_full = w * factor
        wh = w_full.to(torch.float16)
        wl = (w_full - wh.to(torch.float32)).to(torch.float16)
        if not (torch.isfinite(wh).all() and torch.isfinite(wl).all()):
            return
        if b is None:
            bias = torch.zeros(w.shape[0], device=w.device, dtype=torch.float32)
        else:
            bias = (b * factor).to(torch.float32)
        self._wh = wh.contiguous()
        self._wl = wl.contiguous()
        self._bias = bias.contiguous()
        self._K = w.shape[1]

    def _forward_ref(self, x):
        y = F.linear(x, self.matmul.weight, self.matmul.bias)
        original_x = y.clone().detach()
        y = y * self.scaling_factor
        y = y + original_x
        return y

    def forward(self, x):
        if (
            isinstance(x, torch.Tensor)
            and x.is_cuda
            and x.dtype == torch.float32
            and x.dim() == 2
            and not torch.is_grad_enabled()
        ):
            try:
                if not _kernel_ready[0]:
                    _ensure_kernel()
                if _kernel_ready[0]:
                    if self._wh is None or self._wh.device != x.device:
                        self._prepare()
                    if self._wh is not None and x.shape[1] == self._K:
                        if not x.is_contiguous():
                            x = x.contiguous()
                        return _split_gemm(x, self._wh, self._wl, self._bias)
            except Exception:
                pass
        return self._forward_ref(x)
