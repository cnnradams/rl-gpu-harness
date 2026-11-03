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
    def _split_mm_kernel_v5(AH, AL, BH, BL, BIAS, C, M, N, K,
                            sam, sak, sbk, sbn, scm, scn,
                            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                            GROUP: tl.constexpr):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BM)
        num_pid_n = tl.cdiv(N, BN)
        num_pid_in_group = GROUP * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP
        group_size_m = min(num_pid_m - first_pid_m, GROUP)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        AHp = AH + rm[:, None] * sam + rk[None, :] * sak
        ALp = AL + rm[:, None] * sam + rk[None, :] * sak
        BHp = BH + rk[:, None] * sbk + rn[None, :] * sbn
        BLp = BL + rk[:, None] * sbk + rn[None, :] * sbn
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, tl.cdiv(K, BK)):
            ah = tl.load(AHp)
            al = tl.load(ALp)
            bh = tl.load(BHp)
            bl = tl.load(BLp)
            acc = tl.dot(ah, bh, acc)
            acc = tl.dot(al, bh, acc)
            acc = tl.dot(ah, bl, acc)
            AHp += BK * sak
            ALp += BK * sak
            BHp += BK * sbk
            BLp += BK * sbk
        bias = tl.load(BIAS + rn)
        acc = acc + bias[None, :]
        Cp = C + rm[:, None] * scm + rn[None, :] * scn
        tl.store(Cp, acc)


def _split_linear(x, w_hi, w_lo, bias):
    M, K = x.shape
    N = w_hi.shape[1]
    x_hi = x.to(torch.bfloat16)
    x_lo = (x - x_hi.to(torch.float32)).to(torch.bfloat16)
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    BM, BN, BK, GROUP = 128, 256, 32, 8
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _split_mm_kernel_v5[grid](
        x_hi, x_lo, w_hi, w_lo, bias, out, M, N, K,
        x_hi.stride(0), x_hi.stride(1),
        w_hi.stride(0), w_hi.stride(1),
        out.stride(0), out.stride(1),
        BM=BM, BN=BN, BK=BK, GROUP=GROUP,
        num_warps=8, num_stages=3,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = float(scaling_factor)
        self._w_hi = None
        self._w_lo = None
        self._bias = None
        self._w_fb = None

    def _prepare(self, device):
        if self._w_hi is not None and self._w_hi.device == device:
            return
        s = 1.0 + self.scaling_factor
        W = self.matmul.weight.detach().to(device=device, dtype=torch.float32)
        b = self.matmul.bias.detach().to(device=device, dtype=torch.float32)
        Wt = (W * s).t().contiguous()
        hi = Wt.to(torch.bfloat16)
        lo = (Wt - hi.to(torch.float32)).to(torch.bfloat16)
        self._w_hi = hi
        self._w_lo = lo
        self._bias = (b * s).contiguous()
        self._w_fb = W * s

    def forward(self, x):
        self._prepare(x.device)
        if _HAS_TRITON and x.is_cuda and x.shape[0] % 128 == 0 and x.shape[1] % 32 == 0:
            return _split_linear(x.contiguous(), self._w_hi, self._w_lo, self._bias)
        return torch.nn.functional.linear(x, self._w_fb, self._bias)
