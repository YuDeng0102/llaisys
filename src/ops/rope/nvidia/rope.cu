#include "rope.cuh"

template <typename T>
__global__ void rope_kernel(T *out, T *in, const int64_t *pos_ids, size_t seq_len, size_t heads, size_t d_model, float theta, size_t num_per_thread) {
    extern __shared__ unsigned char shared_mem[];
    T *table_vals = reinterpret_cast<T *>(shared_mem);
    size_t tid = threadIdx.x, bid = blockIdx.x;
// #pragma unroll
    for (size_t i = 0; i < num_per_thread; i++) {
        size_t dim = tid * num_per_thread + i;
        if (dim < d_model) {
            float pos_id = static_cast<float>(pos_ids[bid]);
            float exponent = static_cast<float>(dim) / static_cast<float>(d_model);
            float inv_theta_pow = __frcp_rn(__powf(theta, exponent));  // 1 / theta^exponent
            float angle = pos_id * inv_theta_pow;
            table_vals[dim] = to_cuda_type<T>(sinf(angle));
            table_vals[d_model + dim] = to_cuda_type<T>(cosf(angle));
        }
    }
    __syncthreads();

    for (size_t h = 0; h < heads; h++) {
        for (size_t i = 0; i < num_per_thread; i++) {
            size_t dim = tid * num_per_thread + i;
            if (dim < d_model) {
                T a=in[bid*heads*d_model*2 +h*d_model*2 + dim], b=in[bid*heads*d_model*2 + h*d_model*2 + d_model + dim];
                T sin_val = table_vals[dim];
                T cos_val = table_vals[d_model + dim];
                out[bid*heads*d_model*2 + h*d_model*2 + dim] = a * cos_val - b * sin_val;
                out[bid*heads*d_model*2 + h*d_model*2 + d_model + dim] = b * cos_val + a * sin_val;
            }
        }
    }
}

template <typename T>
void rope_(T *out, T *in, const int64_t *pos_ids, size_t seq_len, size_t heads, size_t d_model, float theta) {
    size_t block_size = 256;
    size_t grid_size = seq_len, num_per_thread = (d_model +block_size - 1) / block_size;
    size_t shared_mem_size = sizeof(T) * d_model * 2;
    rope_kernel<T><<<grid_size, block_size, shared_mem_size>>>(out, in, pos_ids, seq_len, heads, d_model, theta, num_per_thread);
}

namespace llaisys::ops::nvidia {
void rope(std::byte *out, std::byte *in, llaisysDataType_t dtype, std::byte *pos_ids, size_t seq_len, size_t heads, size_t d_model, float theta) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        rope_<float>(reinterpret_cast<float *>(out), reinterpret_cast<float *>(in), reinterpret_cast<int64_t *>(pos_ids), seq_len, heads, d_model, theta);
        break;
    case LLAISYS_DTYPE_F16:
        rope_<__half>(reinterpret_cast<__half *>(out), reinterpret_cast<__half *>(in), reinterpret_cast<int64_t *>(pos_ids), seq_len, heads, d_model, theta);
        break;
    case LLAISYS_DTYPE_BF16:
        rope_<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16 *>(out), reinterpret_cast<__nv_bfloat16 *>(in), reinterpret_cast<int64_t *>(pos_ids), seq_len, heads, d_model, theta);
        break;
    default:
        throw std::runtime_error("rope: dtype not supported");
    }
}
} // namespace llaisys::ops::nvidia