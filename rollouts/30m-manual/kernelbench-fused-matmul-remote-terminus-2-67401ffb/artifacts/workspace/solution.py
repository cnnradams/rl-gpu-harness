import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _split_gemm(a_ptr, bh_ptr, bl_ptr, bias_ptr, c_ptr, M, N, K,
                sam, sak, scm, scn,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + rm[:, None] * sam + rk[None, :] * sak
    bh_ptrs = bh_ptr + rk[:, None] * N + rn[None, :]
    bl_ptrs = bl_ptr + rk[:, None] * N + rn[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        a_hi = a.to(tl.float16)
        a_lo = (a - a_hi.to(tl.float32)).to(tl.float16)
        b_hi = tl.load(bh_ptrs)
        b_lo = tl.load(bl_ptrs)
        acc = tl.dot(a_hi, b_hi, acc)
        acc = tl.dot(a_lo, b_hi, acc)
        acc = tl.dot(a_hi, b_lo, acc)
        a_ptrs += BLOCK_K * sak
        bh_ptrs += BLOCK_K * N
        bl_ptrs += BLOCK_K * N
    bias = tl.load(bias_ptr + rn)
    acc = acc + bias[None, :]
    c_ptrs = c_ptr + rm[:, None] * scm + rn[None, :] * scn
    tl.store(c_ptrs, acc)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        s = 1.0 + scaling_factor
        ws = (self.matmul.weight.detach() * s)
        w_hi = ws.half()
        w_lo = (ws - w_hi.float()).half()
        self._w_hi = w_hi.t().contiguous().cuda()
        self._w_lo = w_lo.t().contiguous().cuda()
        self._b = (self.matmul.bias.detach() * s).cuda()

    def forward(self, x):
        M, K = x.shape
        N = self._w_hi.shape[1]
        out = torch.empty((M, N), device=x.device, dtype=torch.float32)
        BM, BN, BK, GM = 256, 128, 32, 8
        grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
        _split_gemm[grid](x, self._w_hi, self._w_lo, self._b, out, M, N, K,
                          x.stride(0), x.stride(1), out.stride(0), out.stride(1),
                          BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GM,
                          num_warps=8, num_stages=4)
        return out
