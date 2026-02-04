#include "../../../utils.hpp"
#include "argmax_cuda.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <limits>

namespace {

// 获取类型的最小值
template <typename T>
__device__ __forceinline__ T get_lowest() {
    if constexpr (std::is_same_v<T, float>) {
        return -INFINITY;
    } else if constexpr (std::is_same_v<T, __half>) {
        return __float2half(-65504.0f); // FP16 最小值
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __float2bfloat16(-3.38953e38f); // BF16 近似最小值
    }
    return T();
}

template <typename T, const size_t block_size>
__global__ void argmax_kernel(const T *vals, size_t *max_idx, size_t *max_idx_per_block, T *max_val, T *max_val_per_block, size_t numel) {
    __shared__ T shared_vals[block_size];
    __shared__ size_t shared_idxs[block_size];
    size_t idx = blockIdx.x * block_size + threadIdx.x, tid = threadIdx.x;
    if (idx < numel) {
        shared_vals[tid] = vals[idx];
        shared_idxs[tid] = idx;
    }else {
        shared_vals[tid] = get_lowest<T>();
        shared_idxs[tid] = 0;
    }
    __syncthreads();

    for (size_t offset = block_size / 2; offset > 0; offset >>= 1) {
        if (tid < offset && idx + offset < numel && shared_vals[tid] < shared_vals[tid + offset]) {
            shared_vals[tid] = shared_vals[tid + offset];
            shared_idxs[tid] = shared_idxs[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        max_idx_per_block[blockIdx.x] = shared_idxs[0];
        max_val_per_block[blockIdx.x] = shared_vals[0];
    }
}

template <typename T, const size_t block_size>
__global__ void final_argmax_kernel(const size_t *max_idx_per_block, const T *max_val_per_block, size_t *max_idx, T *max_val, size_t num_blocks) {
    __shared__ T shared_vals[block_size];
    __shared__ size_t shared_idxs[block_size];
    size_t tid = threadIdx.x;
    if(tid<num_blocks) {
        shared_vals[tid] = max_val_per_block[tid];
        shared_idxs[tid] = max_idx_per_block[tid];
    } else {
        shared_vals[tid] = get_lowest<T>();
        shared_idxs[tid] = 0;
    }
    for (size_t offset = block_size / 2; offset > 0; offset >>= 1) {
        if (tid < offset && shared_vals[tid] < shared_vals[tid + offset]) {

            shared_vals[tid] = shared_vals[tid + offset];
            shared_idxs[tid] = shared_idxs[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        max_idx[0] = shared_idxs[0];
        max_val[0] = shared_vals[0];
    }
}

template <typename T>
void argmax_(const T *vals, size_t *max_idx, T *max_val, size_t numel) {
    if (numel == 0)
        return;

    size_t block_size = 512;
    size_t num_blocks = (numel + block_size - 1) / block_size;
    
    size_t *max_idx_per_block;
    T *max_val_per_block;
    CHECK_CUDA(cudaMalloc(&max_idx_per_block, num_blocks * sizeof(size_t)));
    CHECK_CUDA(cudaMalloc(&max_val_per_block, num_blocks * sizeof(T)));

    argmax_kernel<T, 512><<<num_blocks, block_size>>>(vals, max_idx, max_idx_per_block, max_val, max_val_per_block, numel);
    final_argmax_kernel<T, 512><<<1, (num_blocks < block_size ? num_blocks : block_size)>>>(max_idx_per_block, max_val_per_block, max_idx, max_val, num_blocks);
    
    CHECK_CUDA(cudaFree(max_idx_per_block));
    CHECK_CUDA(cudaFree(max_val_per_block));
    CHECK_CUDA(cudaDeviceSynchronize());
}

} // namespace

namespace llaisys::ops::nvidia {
void argmax(std::byte *vals, std::byte *max_idx, std::byte *max_val, llaisysDataType_t dtype, size_t numel) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return argmax_(reinterpret_cast<const float *>(vals), reinterpret_cast<size_t *>(max_idx), reinterpret_cast<float *>(max_val), numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(reinterpret_cast<const __nv_bfloat16 *>(vals), reinterpret_cast<size_t *>(max_idx), reinterpret_cast<__nv_bfloat16 *>(max_val), numel);
    case LLAISYS_DTYPE_F16:
        return argmax_(reinterpret_cast<const __half *>(vals), reinterpret_cast<size_t *>(max_idx), reinterpret_cast<__half *>(max_val), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::nvidia
