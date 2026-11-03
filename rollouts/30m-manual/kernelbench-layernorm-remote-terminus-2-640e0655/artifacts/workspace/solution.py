import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#define FULL_MASK 0xffffffffu

__inline__ __device__ float warp_sum(float v) {
  v += __shfl_xor_sync(FULL_MASK, v, 16);
  v += __shfl_xor_sync(FULL_MASK, v, 8);
  v += __shfl_xor_sync(FULL_MASK, v, 4);
  v += __shfl_xor_sync(FULL_MASK, v, 2);
  v += __shfl_xor_sync(FULL_MASK, v, 1);
  return v;
}

__global__ void row_partials(const float* __restrict__ x, float* __restrict__ partials,
                             long D, int nblocks) {
  const int row = blockIdx.y;
  const float4* xr = reinterpret_cast<const float4*>(x + (long)row * D);
  const long n4 = D >> 2;
  float s = 0.f, sq = 0.f;
  for (long i = (long)blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x; i < n4; i += (long)gridDim.x * blockDim.x * blockDim.y) {
    float4 v = xr[i];
    s += v.x + v.y + v.z + v.w;
    sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }
  s = warp_sum(s); sq = warp_sum(sq);
  __shared__ float sh_s[32];
  __shared__ float sh_q[32];
  const int lane = threadIdx.x, warp = threadIdx.y;
  const int nw = blockDim.y;
  if (lane == 0) { sh_s[warp] = s; sh_q[warp] = sq; }
  __syncthreads();
  if (warp == 0) {
    float a = (lane < nw) ? sh_s[lane] : 0.f;
    float b = (lane < nw) ? sh_q[lane] : 0.f;
    a = warp_sum(a); b = warp_sum(b);
    if (lane == 0) {
      partials[(long)row * 2 * nblocks + blockIdx.x] = a;
      partials[(long)row * 2 * nblocks + nblocks + blockIdx.x] = b;
    }
  }
}

__global__ void ln_apply(const float* __restrict__ x, const float* __restrict__ w,
                         const float* __restrict__ b, float* __restrict__ y,
                         const float* __restrict__ partials, long D, int nblocks,
                         float eps) {
  const int row = blockIdx.y;
  float s = 0.f, q = 0.f;
  for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < nblocks; i += blockDim.x * blockDim.y) {
    s += partials[(long)row * 2 * nblocks + i];
    q += partials[(long)row * 2 * nblocks + nblocks + i];
  }
  s = warp_sum(s); q = warp_sum(q);
  __shared__ float sh_s[32];
  __shared__ float sh_q[32];
  __shared__ float sh_mean, sh_rstd;
  const int lane = threadIdx.x, warp = threadIdx.y;
  const int nw = blockDim.y;
  if (lane == 0) { sh_s[warp] = s; sh_q[warp] = q; }
  __syncthreads();
  if (warp == 0) {
    float a = (lane < nw) ? sh_s[lane] : 0.f;
    float bb = (lane < nw) ? sh_q[lane] : 0.f;
    a = warp_sum(a); bb = warp_sum(bb);
    if (lane == 0) {
      float mean = a / (float)D;
      float var = bb / (float)D - mean * mean;
      sh_mean = mean;
      sh_rstd = rsqrtf(var + eps);
    }
  }
  __syncthreads();
  const float mean = sh_mean, rstd = sh_rstd;
  const long n4 = D >> 2;
  const float4* xr = reinterpret_cast<const float4*>(x + (long)row * D);
  float4* yr = reinterpret_cast<float4*>(y + (long)row * D);
  const float4* wr = reinterpret_cast<const float4*>(w);
  const float4* br = reinterpret_cast<const float4*>(b);
  for (long i = (long)blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x; i < n4; i += (long)gridDim.x * blockDim.x * blockDim.y) {
    float4 v = xr[i];
    float4 wv = wr[i];
    float4 bv = br[i];
    float4 o;
    o.x = (v.x - mean) * rstd * wv.x + bv.x;
    o.y = (v.y - mean) * rstd * wv.y + bv.y;
    o.z = (v.z - mean) * rstd * wv.z + bv.z;
    o.w = (v.w - mean) * rstd * wv.w + bv.w;
    yr[i] = o;
  }
}

torch::Tensor ln_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32);
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() && bias.is_contiguous());
  auto y = torch::empty_like(x);
  long D = weight.numel();
  TORCH_CHECK((D & 3) == 0);
  long B = x.numel() / D;
  int nb1 = 128, nb2 = 64;
  auto partials = torch::empty({B * nb1 * 2}, x.options());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  dim3 g1(nb1, (unsigned)B), b1(32, 4);
  row_partials<<<g1, b1, 0, stream>>>(x.data_ptr<float>(), partials.data_ptr<float>(), D, nb1);
  dim3 g2(nb2, (unsigned)B), b2(32, 4);
  ln_apply<<<g2, b2, 0, stream>>>(x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
                                  y.data_ptr<float>(), partials.data_ptr<float>(), D, nb1, (float)eps);
  return y;
}
"""

_cpp_src = "torch::Tensor ln_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps);"

_ext = None
_ext_err = None
try:
    _ext = load_inline(
        name="fused_ln_v3",
        cpp_sources=_cpp_src,
        cuda_sources=_cuda_src,
        functions=["ln_forward"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        extra_cflags=["-O3"],
        verbose=False,
    )
except Exception as e:  # pragma: no cover
    _ext_err = e


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.normalized_shape = tuple(normalized_shape)
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _ext is not None and x.is_cuda and x.dtype == torch.float32 and x.is_contiguous():
            return _ext.ln_forward(x, self.weight, self.bias, self.eps)
        return torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
