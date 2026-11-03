import os
import torch
import torch.nn as nn

_CUDA_SRC = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

#define THREADS 256

__global__ void ln_stats_kernel(const float4* __restrict__ x,
                                float* __restrict__ psum,
                                float* __restrict__ psq,
                                long chunk4) {
    long tid = threadIdx.x;
    long base = (long)blockIdx.x * chunk4;
    float s = 0.f, q = 0.f;
    #pragma unroll 4
    for (long i = tid; i < chunk4; i += THREADS) {
        float4 v = x[base + i];
        s += v.x + v.y + v.z + v.w;
        q += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
    __shared__ float sh_s[THREADS], sh_q[THREADS];
    sh_s[tid] = s; sh_q[tid] = q;
    __syncthreads();
    for (int o = THREADS/2; o > 0; o >>= 1) {
        if (tid < o) { sh_s[tid] += sh_s[tid+o]; sh_q[tid] += sh_q[tid+o]; }
        __syncthreads();
    }
    if (tid == 0) { psum[blockIdx.x] = sh_s[0]; psq[blockIdx.x] = sh_q[0]; }
}

__global__ void ln_norm_kernel(const float4* __restrict__ x,
                               float4* __restrict__ y,
                               const float4* __restrict__ w,
                               const float4* __restrict__ b,
                               const float* __restrict__ psum,
                               const float* __restrict__ psq,
                               long chunk4, long N, float eps, int S) {
    int row = blockIdx.x / S;
    int split = blockIdx.x % S;
    float total = 0.f, total2 = 0.f;
    for (int i = 0; i < S; i++) {
        total += psum[row*S + i];
        total2 += psq[row*S + i];
    }
    float mean = total / (float)N;
    float var = total2 / (float)N - mean*mean;
    float rstd = 1.0f / sqrtf(var + eps);
    long tid = threadIdx.x;
    long base = (long)blockIdx.x * chunk4;
    long colbase = (long)split * chunk4;
    #pragma unroll 4
    for (long i = tid; i < chunk4; i += THREADS) {
        float4 v = x[base + i];
        float4 wv = w[colbase + i];
        float4 bv = b[colbase + i];
        float4 o;
        o.x = (v.x - mean) * rstd * wv.x + bv.x;
        o.y = (v.y - mean) * rstd * wv.y + bv.y;
        o.z = (v.z - mean) * rstd * wv.z + bv.z;
        o.w = (v.w - mean) * rstd * wv.w + bv.w;
        y[base + i] = o;
    }
}

torch::Tensor ln_forward(torch::Tensor x, torch::Tensor w, torch::Tensor b,
                         double eps, int64_t S) {
    auto y = torch::empty_like(x);
    long N = w.numel();
    long total = x.numel();
    int B = (int)(total / N);
    long chunk = N / S;
    long chunk4 = chunk / 4;
    int grid = B * (int)S;
    auto psum = torch::empty({grid}, x.options());
    auto psq = torch::empty({grid}, x.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    ln_stats_kernel<<<grid, THREADS, 0, stream>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        psum.data_ptr<float>(), psq.data_ptr<float>(), chunk4);
    ln_norm_kernel<<<grid, THREADS, 0, stream>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(y.data_ptr<float>()),
        reinterpret_cast<const float4*>(w.data_ptr<float>()),
        reinterpret_cast<const float4*>(b.data_ptr<float>()),
        psum.data_ptr<float>(), psq.data_ptr<float>(),
        chunk4, N, (float)eps, (int)S);
    return y;
}
'''

_CPP_SRC = '#include <torch/extension.h>\ntorch::Tensor ln_forward(torch::Tensor x, torch::Tensor w, torch::Tensor b, double eps, int64_t S);'

_ext = None
try:
    from torch.utils.cpp_extension import load_inline
    os.makedirs('/workspace/ln_ext', exist_ok=True)
    _ext = load_inline(
        name='ln_fast_ext',
        cpp_sources=[_CPP_SRC],
        cuda_sources=['#include <ATen/cuda/CUDAContext.h>\n' + _CUDA_SRC],
        functions=['ln_forward'],
        extra_cuda_cflags=['-O3'],
        build_directory='/workspace/ln_ext',
        verbose=False,
    )
except Exception:
    _ext = None


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.N = 1
        for d in self.normalized_shape:
            self.N *= d
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        self.S = 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (_ext is not None and x.is_cuda and x.is_contiguous()
                and x.dtype == torch.float32
                and self.N % (4 * self.S) == 0):
            return _ext.ln_forward(x, self.weight, self.bias, self.eps, self.S)
        return nn.functional.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps)
