#include "../../../utils.hpp"
#include "add_nvidia.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t numel) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < numel; i += stride) {
    c[i] = a[i] + b[i];
  }
}

namespace llaisys::ops::nvidia {
void add(std::byte *c, const std::byte *a, const std::byte *b,
         llaisysDataType_t type, size_t numel) {
  switch (type) {
  case LLAISYS_DTYPE_F32:
    return add_kernel<<<64, 256>>>(reinterpret_cast<float *>(c),
                                   reinterpret_cast<const float *>(a),
                                   reinterpret_cast<const float *>(b), numel);
  case LLAISYS_DTYPE_BF16:
    return add_kernel<<<64, 256>>>(reinterpret_cast<__nv_bfloat16 *>(c),
                                   reinterpret_cast<const __nv_bfloat16 *>(a),
                                   reinterpret_cast<const __nv_bfloat16 *>(b),
                                   numel);
  case LLAISYS_DTYPE_F16:
    return add_kernel<<<64, 256>>>(reinterpret_cast<__half *>(c),
                                   reinterpret_cast<const __half *>(a),
                                   reinterpret_cast<const __half *>(b), numel);
  default:
    EXCEPTION_UNSUPPORTED_DATATYPE(type);
  }
}
} // namespace llaisys::ops::nvidia
