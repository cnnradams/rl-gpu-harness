import torch
import torch.nn as nn

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


if _HAS_TRITON:
    @triton.jit
    def _mm_fp16x3_kernel(
        a_ptr, bh_ptr, bl_ptr, bias_ptr, c_ptr,
        M, N, K,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BM)
        num_pid_n = tl.cdiv(N, BN)
        num_pid_in_group = GM * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GM
        group_size_m = min(num_pid_m - first_pid_m, GM)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        a_ptrs = a_ptr + rm[:, None] * K + rk[None, :]
        bh_ptrs = bh_ptr + rk[:, None] * N + rn[None, :]
        bl_ptrs = bl_ptr + rk[:, None] * N + rn[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, K, BK):
            a = tl.load(a_ptrs)
            ah = a.to(tl.float16)
            al = (a - ah.to(tl.float32)).to(tl.float16)
            bh = tl.load(bh_ptrs)
            bl = tl.load(bl_ptrs)
            acc = tl.dot(al, bh, acc=acc)
            acc = tl.dot(ah, bl, acc=acc)
            acc = tl.dot(ah, bh, acc=acc)
            a_ptrs += BK
            bh_ptrs += BK * N
            bl_ptrs += BK * N
        bias = tl.load(bias_ptr + rn).to(tl.float32)
        acc = acc + bias[None, :]
        tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc)


def _triton_mm(a, bh, bl, bias, M, N, K):
    out = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, 256) * triton.cdiv(N, 128),)
    _mm_fp16x3_kernel[grid](
        a, bh, bl, bias, out, M, N, K,
        BM=256, BN=128, BK=32, GM=8, num_warps=8, num_stages=3)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self._key = None

    def _prepare(self, x):
        with torch.no_grad():
            alpha = 1.0 + float(self.scaling_factor)
            B = (self.matmul.weight.detach().t().contiguous() * alpha).contiguous()
            self._Bh = B.half()
            self._Bl = (B - self._Bh.float()).half()
            self._biasv = (self.matmul.bias.detach() * alpha).contiguous()
            self._B32 = B
        self._key = (x.device, x.dtype)

    def forward(self, x):
        if self._key != (x.device, x.dtype):
            self._prepare(x)
        M, K = x.shape
        N = self._Bh.shape[1]
        if (_HAS_TRITON and x.is_cuda and x.dtype == torch.float32
                and x.is_contiguous()
                and M % 256 == 0 and N % 128 == 0 and K % 32 == 0):
            return _triton_mm(x, self._Bh, self._Bl, self._biasv, M, N, K)
        return torch.addmm(self._biasv, x, self._B32)
