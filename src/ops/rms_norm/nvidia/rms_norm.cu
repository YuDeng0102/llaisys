#include "rms_norm.cuh"


constexpr int WARP_SIZE = 32;  // NVIDIA GPU warp size


template <typename T, const int kWarpSize = 32>
__device__ __forceinline__ T warpReduceSum(T val) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename T, const int kBlockSize = 256>
__device__ T blockReduceSum(T val) {
    constexpr int numWarps = (kBlockSize + 31) / 32;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    __shared__ T shared[numWarps];
    val = warpReduceSum<T>(val);
    if (lane_id == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    val = lane_id < numWarps ? shared[lane_id] : T(0);
    val = warpReduceSum<T, numWarps>(val);
    return val;
}

template <typename T>
__global__ void rms_norm_2d_kernel(T *out, const T *in, const T *weight, float eps, size_t dim0, size_t dim1, const size_t nums_per_thread) {
    __shared__ T s_variance;
    size_t bid = blockIdx.x, tid = threadIdx.x;
    T variance = T(0);
    // Compute sum of squares
    for (size_t i = 0; i < nums_per_thread; i++) {
        size_t j = tid * nums_per_thread + i;
        if (j >= dim1) {
            break;
        }
        variance += in[bid * dim1 + j] * in[bid * dim1 + j];
    }
    // warp reduce sum_sq
    variance = blockReduceSum<T>(variance);
    if (tid == 0) {
        s_variance = rsqrt_unified(variance / to_cuda_type<T>(static_cast<float>(dim1)) + to_cuda_type<T>(eps));
    }
    __syncthreads();
    for( size_t i = 0; i < nums_per_thread; i++) {
        size_t j = tid * nums_per_thread + i;
        if (j >= dim1) {
            break;
        }
        out[bid * dim1 + j] = in[bid * dim1 + j] * s_variance * weight[j];
    }
}

template <typename T>
void rms_norm_2d_(T *out, const T *in, const T *weight, float eps, size_t dim0, size_t dim1) {
    size_t BLOCK_SIZE = 256;
    size_t BLOCKS = dim0, nums_per_thread = (dim1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    rms_norm_2d_kernel<T><<<BLOCKS, BLOCK_SIZE>>>(out, in, weight, eps, dim0, dim1, nums_per_thread);
}

namespace llaisys::ops::nvidia {
void rms_norm_2d(std::byte *out, std::byte *in, std::byte *weight, float eps, llaisysDataType_t dtype, size_t dim0, size_t dim1) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_2d_<float>(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), eps, dim0, dim1);
    case LLAISYS_DTYPE_F16:
        return rms_norm_2d_<__half>(reinterpret_cast<__half *>(out), reinterpret_cast<const __half *>(in), reinterpret_cast<const __half *>(weight), eps, dim0, dim1);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_2d_<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16 *>(out), reinterpret_cast<const __nv_bfloat16 *>(in), reinterpret_cast<const __nv_bfloat16 *>(weight), eps, dim0, dim1);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
