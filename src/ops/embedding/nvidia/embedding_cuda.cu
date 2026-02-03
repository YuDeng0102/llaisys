#include "../../../utils.hpp"
#include "embedding_cuda.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// CUDA kernel for embedding lookup
template <typename T>
__global__ void embedding_kernel(T *out, const int64_t *index, const T *weight, 
                                  size_t batch_size, size_t seq_len, size_t vocab_size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch_size * seq_len;
    
    if (idx < total) {
        size_t batch_id = idx / seq_len;
        size_t pos_in_seq = idx % seq_len;
        
        // Get the vocabulary index for this batch
        int64_t vocab_idx = index[batch_id];
        
        // Bounds check
        if (vocab_idx >= 0 && vocab_idx < (int64_t)vocab_size) {
            // Copy embedding from weight to output
            // weight shape: (vocab_size, seq_len)
            // out shape: (batch_size, seq_len)
            out[idx] = weight[vocab_idx * seq_len + pos_in_seq];
        }
    }
}

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight, 
                size_t batch_size, size_t seq_len, size_t vocab_size) {
    size_t total_elements = batch_size * seq_len;
    size_t block_size = 256;
    size_t num_blocks = (total_elements + block_size - 1) / block_size;
    
    embedding_kernel<T><<<num_blocks, block_size>>>(
        out, index, weight, batch_size, seq_len, vocab_size
    );
    CHECK_CUDA(cudaDeviceSynchronize());
}

namespace llaisys::ops::nvidia {
void embedding(std::byte *out, std::byte *index, std::byte *weight, llaisysDataType_t dtype, 
               size_t batch_size, size_t seq_len, size_t vocab_size) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        embedding_(reinterpret_cast<float *>(out), reinterpret_cast<const int64_t *>(index),
                   reinterpret_cast<const float *>(weight), batch_size, seq_len, vocab_size);
        break;
    case LLAISYS_DTYPE_F16:
        embedding_(reinterpret_cast<__half *>(out), reinterpret_cast<const int64_t *>(index),
                   reinterpret_cast<const __half *>(weight), batch_size, seq_len, vocab_size);
        break;
    case LLAISYS_DTYPE_BF16:
        embedding_(reinterpret_cast<__nv_bfloat16 *>(out), reinterpret_cast<const int64_t *>(index),
                   reinterpret_cast<const __nv_bfloat16 *>(weight), batch_size, seq_len, vocab_size);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::nvidia
