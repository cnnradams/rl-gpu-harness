import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BM': 256, 'BN': 128, 'BK': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BM': 256, 'BN': 128, 'BK': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _split_gemm_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    M, N, K,
    sxm, sxk, swk, swn, sym, syn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GROUP_M: tl.constexpr, SCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    mask_m = rm < M
    mask_n = rn < N

    x_ptrs = x_ptr + rm[:, None] * sxm + rk[None, :] * sxk
    w_ptrs = w_ptr + rk[:, None] * swk + rn[None, :] * swn

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        mk = mask_m[:, None] & (rk[None, :] + k * BK < K)
        x = tl.load(x_ptrs, mask=mk, other=0.0)
        nk = (rk[:, None] + k * BK < K) & mask_n[None, :]
        w = tl.load(w_ptrs, mask=nk, other=0.0)
        x_hi = x.to(tl.float16)
        x_lo = (x - x_hi.to(tl.float32)).to(tl.float16)
        w_hi = w.to(tl.float16)
        w_lo = (w - w_hi.to(tl.float32)).to(tl.float16)
        acc = tl.dot(x_hi, w_hi, acc)
        acc = tl.dot(x_hi, w_lo, acc)
        acc = tl.dot(x_lo, w_hi, acc)
        x_ptrs += BK * sxk
        w_ptrs += BK * swk

    bias = tl.load(b_ptr + rn, mask=mask_n, other=0.0)
    acc = acc * SCALE + (SCALE * bias)[None, :]
    y_ptrs = y_ptr + rm[:, None] * sym + rn[None, :] * syn
    tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def _split_linear(x, w_t, b, scale):
    M, K = x.shape
    N = w_t.shape[1]
    y = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(M, META['BM']) * triton.cdiv(N, META['BN']),)
    _split_gemm_kernel[grid](
        x, w_t, b, y, M, N, K,
        x.stride(0), x.stride(1), w_t.stride(0), w_t.stride(1), y.stride(0), y.stride(1),
        SCALE=scale,
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self._wt = None
        self._key = None

    def _prep(self):
        w = self.matmul.weight
        key = (w.data_ptr(), w.device, w.dtype)
        if self._key != key or self._wt is None:
            self._wt = w.detach().t().contiguous()
            self._key = key
        return self._wt

    def forward(self, x):
        w_t = self._prep()
        b = self.matmul.bias
        if x.is_cuda and x.dtype == torch.float32 and w_t.dtype == torch.float32:
            return _split_linear(x.contiguous(), w_t, b, float(self.scaling_factor))
        y = torch.nn.functional.linear(x, self.matmul.weight, b)
        return y * self.scaling_factor + y
