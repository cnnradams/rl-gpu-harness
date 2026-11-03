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
    def _linear_scaled_kernel(
        a_ptr, wh_ptr, wl_ptr, b_ptr, c_ptr,
        M, N, K,
        scale,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
        wh_ptrs = wh_ptr + offs_k[:, None] * N + offs_n[None, :]
        wl_ptrs = wl_ptr + offs_k[:, None] * N + offs_n[None, :]

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs)
            a_hi = a.to(tl.float16)
            a_lo = (a - a_hi.to(tl.float32)).to(tl.float16)
            whl = tl.load(wh_ptrs)
            wll = tl.load(wl_ptrs)
            acc = tl.dot(a_lo, whl, acc)
            acc = tl.dot(a_hi, wll, acc)
            acc = tl.dot(a_hi, whl, acc)
            a_ptrs += BLOCK_K
            wh_ptrs += BLOCK_K * N
            wl_ptrs += BLOCK_K * N

        bias = tl.load(b_ptr + offs_n)
        acc = (acc + bias[None, :]) * scale
        c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
        tl.store(c_ptrs, acc)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self._w_hi = None
        self._w_lo = None
        self._w_key = None

    def _get_wsplit(self, w):
        key = (w.data_ptr(), w._version, w.device.index)
        if self._w_key != key:
            wt = w.detach().t().contiguous()
            wh = wt.to(torch.float16)
            wl = (wt - wh.to(torch.float32)).to(torch.float16)
            self._w_hi = wh
            self._w_lo = wl
            self._w_key = key
        return self._w_hi, self._w_lo

    def forward(self, x):
        w = self.matmul.weight
        b = self.matmul.bias
        if (
            _HAS_TRITON
            and x.is_cuda
            and x.dtype == torch.float32
            and w.dtype == torch.float32
            and x.dim() == 2
            and x.shape[1] == w.shape[1]
            and x.is_contiguous()
            and x.shape[0] % 256 == 0
            and w.shape[0] % 128 == 0
            and w.shape[1] % 32 == 0
        ):
            M, K = x.shape
            N = w.shape[0]
            wh, wl = self._get_wsplit(w)
            out = torch.empty((M, N), device=x.device, dtype=torch.float32)
            scale = float(1.0 + self.scaling_factor)
            grid = ((M // 256) * (N // 128),)
            _linear_scaled_kernel[grid](
                x, wh, wl, b, out,
                M, N, K,
                scale,
                BLOCK_M=256, BLOCK_N=128, BLOCK_K=32, GROUP_M=8,
                num_stages=5, num_warps=8,
            )
            return out
        t = F.linear(x, w, b)
        return t * self.scaling_factor + t
