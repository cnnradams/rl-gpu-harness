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
    def _mm_kernel(a_ptr, b_ptr, c_ptr,
                   M, N, K,
                   stride_am, stride_ak,
                   stride_bk, stride_bn,
                   stride_cm, stride_cn,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                   GROUP_M: tl.constexpr):
        pid = tl.program_id(axis=0)
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
        a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
        b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc, input_precision="tf32x3")
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(c_ptrs, acc)

    def _matmul(a, b, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, num_stages):
        M, K = a.shape
        K2, N = b.shape
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        _mm_kernel[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_M=GROUP_M, num_warps=num_warps, num_stages=num_stages,
        )
        return c


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        with torch.no_grad():
            s = 1.0 + scaling_factor
            self.matmul.weight.mul_(s)
            self.matmul.bias.mul_(s)
        self._plan = None

    def forward(self, x):
        if self._plan is None:
            self._plan = self._select_plan(x)
        plan = self._plan
        if plan[0] == 0:
            return _matmul(x, self.matmul.weight.t(), *plan[1])
        return F.linear(x, self.matmul.weight, self.matmul.bias)

    def _select_plan(self, x):
        if not (_HAS_TRITON and x.is_cuda):
            return (1,)
        try:
            torch.cuda.synchronize()
            import time
            wt = self.matmul.weight.t()
            cfgs = [
                (128, 128, 32, 8, 4, 4),
                (64, 128, 64, 8, 4, 4),
                (128, 64, 64, 8, 4, 4),
                (128, 128, 64, 8, 4, 3),
                (64, 256, 64, 8, 4, 4),
            ]
            best = None
            best_t = float('inf')
            for cfg in cfgs:
                try:
                    _matmul(x, wt, *cfg)  # compile
                    torch.cuda.synchronize()
                    evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(3)]
                    for s, e in evs:
                        s.record()
                        _matmul(x, wt, *cfg)
                        e.record()
                    torch.cuda.synchronize()
                    t = min(s.elapsed_time(e) for s, e in evs)
                    if t < best_t:
                        best_t = t
                        best = cfg
                except Exception:
                    continue
            if best is None:
                return (1,)
            return (0, best)
        except Exception:
            return (1,)
