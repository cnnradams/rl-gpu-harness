import torch
import torch.nn as nn
import triton
import triton.language as tl

_alloc_set = False
def _ensure_alloc():
    global _alloc_set
    if not _alloc_set:
        def alloc(size, align, stream):
            return torch.empty(size, dtype=torch.int8, device='cuda')
        triton.set_allocator(alloc)
        _alloc_set = True


@triton.jit
def _split_kernel(x_ptr, hi_ptr, lo_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < numel
    v = tl.load(x_ptr + offs, mask=m)
    hi = v.to(tl.float16)
    lo = (v - hi.to(tl.float32)).to(tl.float16)
    tl.store(hi_ptr + offs, hi, mask=m)
    tl.store(lo_ptr + offs, lo, mask=m)


@triton.jit
def _gemm_f16x3_ws(ah_ptr, al_ptr, bh_ptr, bl_ptr, c_ptr, Bias, M, N, K,
                   ALPHA: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                   GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr):
    desc_ah = tl.make_tensor_descriptor(ah_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K])
    desc_al = tl.make_tensor_descriptor(al_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K])
    desc_bh = tl.make_tensor_descriptor(bh_ptr, shape=[K, N], strides=[N, 1], block_shape=[BLOCK_K, BLOCK_N])
    desc_bl = tl.make_tensor_descriptor(bl_ptr, shape=[K, N], strides=[N, 1], block_shape=[BLOCK_K, BLOCK_N])
    desc_c = tl.make_tensor_descriptor(c_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N])
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n

    for tile_id in tl.range(tl.program_id(0), num_tiles, NUM_SMS, warp_specialize=True):
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (tile_id % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        offs_am = pid_m * BLOCK_M
        offs_bn = pid_n * BLOCK_N
        acc0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_K
            ah = desc_ah.load([offs_am, offs_k])
            al = desc_al.load([offs_am, offs_k])
            bh = desc_bh.load([offs_k, offs_bn])
            bl = desc_bl.load([offs_k, offs_bn])
            acc0 = tl.dot(ah, bh, acc0)
            acc1 = tl.dot(ah, bl, acc1)
            acc1 = tl.dot(al, bh, acc1)
        acc = acc0 + acc1
        bias = tl.load(Bias + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.float32)
        acc = acc * ALPHA + (ALPHA * bias)[None, :]
        desc_c.store([offs_am, offs_bn], acc)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = float(scaling_factor)
        self._alpha = 1.0 + self.scaling_factor
        self._wh = None
        self._wl = None
        self._cache_key = None
        self._bufs = {}
        self._num_sms = None

    def _get_wsplits(self, w):
        key = (w.device, w.dtype, w.data_ptr(), w._version)
        if self._cache_key != key:
            wt = w.t().contiguous()
            wh = wt.to(torch.float16)
            wl = (wt - wh.to(torch.float32)).to(torch.float16)
            self._wh = wh.contiguous()
            self._wl = wl.contiguous()
            self._cache_key = key
        return self._wh, self._wl

    def forward(self, x):
        w = self.matmul.weight
        b = self.matmul.bias
        if (x.is_cuda and x.dtype == torch.float32 and x.dim() == 2
                and x.stride(1) == 1 and w.stride(1) == 1):
            M, K = x.shape
            N = w.shape[0]
            if M % 128 == 0 and N % 128 == 0 and K % 64 == 0 and K == w.shape[1]:
                _ensure_alloc()
                wh, wl = self._get_wsplits(w)
                bufs = self._bufs.get((x.device, M, N, K))
                if bufs is None:
                    xh = torch.empty((M, K), device=x.device, dtype=torch.float16)
                    xl = torch.empty((M, K), device=x.device, dtype=torch.float16)
                    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
                    self._bufs[(x.device, M, N, K)] = (xh, xl, out)
                else:
                    xh, xl, out = bufs
                numel = x.numel()
                _split_kernel[(triton.cdiv(numel, 16384),)](x, xh, xl, numel, BLOCK=16384, num_warps=8)
                if self._num_sms is None:
                    self._num_sms = torch.cuda.get_device_properties(x.device).multi_processor_count
                num_sms = self._num_sms
                BM, BN, BK, GM = 128, 128, 64, 4
                grid = (min(num_sms, triton.cdiv(M, BM) * triton.cdiv(N, BN)),)
                _gemm_f16x3_ws[grid](xh, xl, wh, wl, out, b, M, N, K,
                                     ALPHA=self._alpha,
                                     BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                                     GROUP_M=GM, NUM_SMS=num_sms,
                                     num_warps=4, num_stages=2)
                return out
        return torch.addmm(b, x, w.t(), alpha=self._alpha)
