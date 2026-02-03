#pragma once

#ifdef ENABLE_NVIDIA_API

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        const cudaError_t error_code = call;                                \
        if (error_code != cudaSuccess) {                                    \
            printf("CUDA Error:\n");                                        \
            printf("    File:   %s\n", __FILE__);                           \
            printf("    Line:   %d\n", __LINE__);                           \
            printf("    Error code: %d\n", error_code);                     \
            printf("    Error text: %s\n", cudaGetErrorString(error_code)); \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

#define CHECK_CUBLAS(call)                          \
    do {                                            \
        const cublasStatus_t status = call;         \
        if (status != CUBLAS_STATUS_SUCCESS) {      \
            printf("cuBLAS Error:\n");              \
            printf("    File:   %s\n", __FILE__);   \
            printf("    Line:   %d\n", __LINE__);   \
            printf("    Error code: %d\n", status); \
            exit(1);                                \
        }                                           \
    } while (0)

// 统一的类型转换
template <typename T>
__device__ __forceinline__ T to_cuda_type(float x) {
    if constexpr (std::is_same_v<T, float>) {
        return x;
    } else if constexpr (std::is_same_v<T, __half>) {
        return __float2half(x);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __float2bfloat16(x);
    }
}

// 转换为 float（反向转换）
template <typename T>
__device__ __forceinline__ float to_float(T x) {
    if constexpr (std::is_same_v<T, float>) {
        return x;
    } else if constexpr (std::is_same_v<T, __half>) {
        return __half2float(x);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __bfloat162float(x);
    }
}

// 统一的 rsqrt 接口
template <typename T>
__device__ __forceinline__ T rsqrt_unified(T x) {
    if constexpr (std::is_same_v<T, float>) {
        return rsqrtf(x);
    } else if constexpr (std::is_same_v<T, double>) {
        return rsqrt(x);
    } else if constexpr (std::is_same_v<T, __half>) {
        return hrsqrt(x);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return hrsqrt(x);
    }
}


#endif
