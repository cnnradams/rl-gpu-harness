import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cuda_src = r'''
#include <cuda_runtime.h>
#include <cstdint>

#define FULL_MASK 0xffffffffu

__global__ void partial_kernel(const float4* __restrict__ x4,
                               float* __restrict__ ps,
                               float* __restrict__ psq,
                               int n4, int chunks) {
    int row = blockIdx.y;
    int chunk = blockIdx.x;
    int chunk_size = (n4 + chunks - 1) / chunks;
    int start = chunk * chunk_size;
    int end = min(start + chunk_size, n4);
    const float4* xr = x4 + (long)row * (long)n4;
    float s = 0.f, sq = 0.f;
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        float4 v = xr[i];
        s += v.x + v.y + v.z + v.w;
        sq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
    __shared__ float sh_s[32];
    __shared__ float sh_q[32];
    for (int off = 16; off > 0; off >>= 1) {
        s += __shfl_down_sync(FULL_MASK, s, off);
        sq += __shfl_down_sync(FULL_MASK, sq, off);
    }
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) { sh_s[warp] = s; sh_q[warp] = sq; }
    __syncthreads();
    int nwarp = (blockDim.x + 31) >> 5;
    if (warp == 0) {
        s = (lane < nwarp) ? sh_s[lane] : 0.f;
        sq = (lane < nwarp) ? sh_q[lane] : 0.f;
        for (int off = 16; off > 0; off >>= 1) {
            s += __shfl_down_sync(FULL_MASK, s, off);
            sq += __shfl_down_sync(FULL_MASK, sq, off);
        }
        if (lane == 0) {
            ps[row * chunks + chunk] = s;
            psq[row * chunks + chunk] = sq;
        }
    }
}

__global__ void norm_kernel(const float4* __restrict__ x4,
                            const float4* __restrict__ w4,
                            const float4* __restrict__ b4,
                            float4* __restrict__ y4,
                            const float* __restrict__ ps,
                            const float* __restrict__ psq,
                            int n4, int chunks, float eps) {
    int row = blockIdx.y;
    int chunk = (int)blockIdx.x;
    int chunk_size = (n4 + chunks - 1) / chunks;
    int start = chunk * chunk_size;
    int end = min(start + chunk_size, n4);
    __shared__ float s_mean, s_rstd;
    if (threadIdx.x == 0) {
        float total = 0.f, totalsq = 0.f;
        for (int c = 0; c < chunks; ++c) {
            total += ps[row * chunks + c];
            totalsq += psq[row * chunks + c];
        }
        float n = (float)n4 * 4.0f;
        float mean = total / n;
        float var = totalsq / n - mean * mean;
        s_mean = mean;
        s_rstd = rsqrtf(var + eps);
    }
    __syncthreads();
    float mean = s_mean;
    float rstd = s_rstd;
    const float4* xr = x4 + (long)row * (long)n4;
    float4* yr = y4 + (long)row * (long)n4;
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        float4 v = xr[i];
        float4 wv = w4[i];
        float4 bv = b4[i];
        float4 o;
        o.x = (v.x - mean) * rstd * wv.x + bv.x;
        o.y = (v.y - mean) * rstd * wv.y + bv.y;
        o.z = (v.z - mean) * rstd * wv.z + bv.z;
        o.w = (v.w - mean) * rstd * wv.w + bv.w;
        yr[i] = o;
    }
}

void ln_two(int64_t x, int64_t w, int64_t b, int64_t y,
            int64_t ps, int64_t psq, int64_t M, int64_t N,
            double eps, int64_t chunks, int64_t blockA, int64_t blockB, int64_t stream) {
    int n4 = (int)(N / 4);
    dim3 grid((unsigned)chunks, (unsigned)M);
    cudaStream_t s = (cudaStream_t)stream;
    partial_kernel<<<grid, (int)blockA, 0, s>>>((const float4*)x, (float*)ps, (float*)psq, n4, (int)chunks);
    norm_kernel<<<grid, (int)blockB, 0, s>>>((const float4*)x, (const float4*)w, (const float4*)b,
                                             (float4*)y, (const float*)ps, (const float*)psq, n4, (int)chunks, (float)eps);
}
'''

_cpp_src = '#include <cstdint>\nvoid ln_two(int64_t x, int64_t w, int64_t b, int64_t y, int64_t ps, int64_t psq, int64_t M, int64_t N, double eps, int64_t chunks, int64_t blockA, int64_t blockB, int64_t stream);'

_mod = load_inline(name='ln_split_eval2', cpp_sources=_cpp_src, cuda_sources=_cuda_src,
                   functions=['ln_two'], extra_cuda_cflags=['-O3'], verbose=False)


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.normalized_shape = tuple(normalized_shape)
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        self.eps = 1e-5
        self._ps = None
        self._psq = None
        self._out = None
        self._chunks = 32

    def _lazy(self, x, lead):
        if self._ps is None or self._ps.device != x.device or self._ps.shape[0] != lead:
            self._ps = torch.empty((lead, self._chunks), device=x.device, dtype=torch.float32)
            self._psq = torch.empty((lead, self._chunks), device=x.device, dtype=torch.float32)
        if self._out is None or self._out.shape != x.shape or self._out.device != x.device:
            self._out = torch.empty_like(x.contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not (x.is_cuda and x.dtype == torch.float32):
            return torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        xs = x.shape
        nn_dim = len(self.normalized_shape)
        lead = 1
        if len(xs) > nn_dim:
            for d in xs[:-nn_dim]:
                lead *= d
        N = 1
        for d in self.normalized_shape:
            N *= d
        xc = x.contiguous()
        self._lazy(xc, lead)
        out = self._out
        stream = torch.cuda.current_stream(x.device).cuda_stream
        w = self.weight.contiguous()
        b = self.bias.contiguous()
        _mod.ln_two(xc.data_ptr(), w.data_ptr(), b.data_ptr(), out.data_ptr(),
                    self._ps.data_ptr(), self._psq.data_ptr(), lead, N,
                    self.eps, self._chunks, 512, 256, stream)
        return out
