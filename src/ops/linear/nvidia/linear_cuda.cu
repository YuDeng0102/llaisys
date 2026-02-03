#include "linear_cuda.cuh"

namespace llaisys::ops::nvidia {

// Kernel to add bias
template <typename T>
__global__ void add_bias_kernel(T *out, const T *bias, size_t batch_size, size_t out_features) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_features)
        return;

    size_t feature_idx = idx % out_features;
    out[idx] += bias[feature_idx];
}

void linear2d(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {

    size_t batch_size = out->shape()[0];
    size_t in_features = in->shape()[1];
    size_t out_features = out->shape()[1];

    void *out_ptr = out->data();
    const void *in_ptr = in->data();
    const void *weight_ptr = weight->data();
    const void *bias_ptr = bias != nullptr ? bias->data() : nullptr;

    // Create cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    try {
        // Matrix multiplication: out = in @ weight^T
        // in: (batch_size, in_features)
        // weight: (out_features, in_features) -> transpose to (in_features, out_features)
        // out: (batch_size, out_features)

        // cuBLAS uses column-major, our data is row-major
        // We want: out = in @ weight^T
        // in: (batch_size, in_features)
        // weight: (out_features, in_features)
        // Equivalent to: out^T = weight @ in^T
        // cuBLAS: C = op(A) @ op(B), so A=weight (no transpose), B=in^T (transpose)

        float alpha = 1.0f;
        float beta = 0.0f;

        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            CHECK_CUBLAS(cublasSgemm(
                handle, CUBLAS_OP_T, CUBLAS_OP_N,
                out_features, batch_size, in_features,
                &alpha,
                reinterpret_cast<const float *>(weight_ptr), in_features,
                reinterpret_cast<const float *>(in_ptr), in_features,
                &beta,
                reinterpret_cast<float *>(out_ptr), out_features));
            break;

        case LLAISYS_DTYPE_F16:
            CHECK_CUBLAS(cublasGemmEx(
                handle, CUBLAS_OP_T, CUBLAS_OP_N,
                out_features, batch_size, in_features,
                &alpha,
                weight_ptr, CUDA_R_16F, in_features,
                in_ptr, CUDA_R_16F, in_features,
                &beta,
                out_ptr, CUDA_R_16F, out_features,
                CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            break;

        case LLAISYS_DTYPE_BF16: {
            // For BF16, use cublasGemmEx for more flexibility
            CHECK_CUBLAS(cublasGemmEx(
                handle, CUBLAS_OP_T, CUBLAS_OP_N,
                out_features, batch_size, in_features,
                &alpha,
                weight_ptr, CUDA_R_16BF, in_features,
                in_ptr, CUDA_R_16BF, in_features,
                &beta,
                out_ptr, CUDA_R_16BF, out_features,
                CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            break;
        }

        default:
            throw std::runtime_error("Unsupported dtype for linear2d");
        }

        // Add bias if provided
        if (bias_ptr != nullptr) {
            int threads_per_block = 256;
            int num_blocks = (batch_size * out_features + threads_per_block - 1) / threads_per_block;

            switch (out->dtype()) {
            case LLAISYS_DTYPE_F32:
                add_bias_kernel<float><<<num_blocks, threads_per_block>>>(
                    reinterpret_cast<float *>(out_ptr),
                    reinterpret_cast<const float *>(bias_ptr),
                    batch_size, out_features);
                break;
            case LLAISYS_DTYPE_F16:
                add_bias_kernel<__half><<<num_blocks, threads_per_block>>>(
                    reinterpret_cast<__half *>(out_ptr),
                    reinterpret_cast<const __half *>(bias_ptr),
                    batch_size, out_features);
                break;
            case LLAISYS_DTYPE_BF16:
                add_bias_kernel<__nv_bfloat16><<<num_blocks, threads_per_block>>>(
                    reinterpret_cast<__nv_bfloat16 *>(out_ptr),
                    reinterpret_cast<const __nv_bfloat16 *>(bias_ptr),
                    batch_size, out_features);
                break;
            default:
                break;
            }
        }

        CHECK_CUDA(cudaDeviceSynchronize());
    } catch (...) {
        CHECK_CUBLAS(cublasDestroy(handle));
        throw;
    }

    CHECK_CUBLAS(cublasDestroy(handle));
}

} // namespace llaisys::ops::nvidia
