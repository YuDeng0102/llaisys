#include "swiglu.cuh"
#include "../../../device/nvidia/common.cuh"

template<typename T>
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        // out[i] = up[i] * gate[i] * sigmoid(gate[i])
        // sigmoid(x) = 1 / (1 + exp(-x))
        
        // Compute in float precision for numerical stability
        float gate_f = to_float(gate[idx]);
        float up_f = to_float(up[idx]);
        
        // Compute sigmoid in float: 1 / (1 + exp(-x))
        float sigmoid_f = __frcp_rn(1.0f + expf(-gate_f));
        
        // SwiGLU: out = up * gate * sigmoid(gate)
        float result = up_f * gate_f * sigmoid_f;
        
        out[idx] = to_cuda_type<T>(result);
    }
}

template<typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t numel) {
    size_t block_size = 256;
    size_t grid_size = (numel + block_size - 1) / block_size;
    
    swiglu_kernel<T><<<grid_size, block_size>>>(out, gate, up, numel);
    CHECK_CUDA(cudaGetLastError());
}

namespace llaisys::ops::nvidia {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        swiglu_(reinterpret_cast<float *>(out), 
                reinterpret_cast<const float *>(gate),
                reinterpret_cast<const float *>(up), 
                numel);
        break;
    case LLAISYS_DTYPE_F16:
        swiglu_(reinterpret_cast<__half *>(out), 
                reinterpret_cast<const __half *>(gate),
                reinterpret_cast<const __half *>(up), 
                numel);
        break;
    case LLAISYS_DTYPE_BF16:
        swiglu_(reinterpret_cast<__nv_bfloat16 *>(out), 
                reinterpret_cast<const __nv_bfloat16 *>(gate),
                reinterpret_cast<const __nv_bfloat16 *>(up), 
                numel);
        break;
    default:
        break;
    }
}
} // namespace llaisys::ops::nvidia
