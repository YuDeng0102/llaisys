#include "embedding_cpu.hpp"
#include "../../../utils.hpp"
#include "llaisys.h"
#include <cstring>
#include <iostream>
#include <stdexcept>

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight, size_t batch_size, size_t seq_len, size_t vocab_size) {

    for (size_t i = 0; i < batch_size; ++i) {
        auto idx = static_cast<size_t>(index[i]);
        if (idx >= vocab_size) {
            throw std::runtime_error("index out of range");
        }
        std::memcpy(out + i * seq_len, weight + idx * seq_len, seq_len * sizeof(T));
    }
}
namespace llaisys::ops::cpu {
void embedding(std::byte *out, std::byte *index, std::byte *weight, llaisysDataType_t dtype, size_t batch_size,
               size_t seq_len, size_t vocab_size) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        embedding_<float>(reinterpret_cast<float *>(out), reinterpret_cast<int64_t *>(index),
                          reinterpret_cast<float *>(weight), batch_size, seq_len, vocab_size);
        break;
    case LLAISYS_DTYPE_F16:
        embedding_<llaisys::fp16_t>(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<int64_t *>(index),
                                    reinterpret_cast<llaisys::fp16_t *>(weight), batch_size, seq_len, vocab_size);
        break;
    case LLAISYS_DTYPE_BF16:
        embedding_<llaisys::bf16_t>(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<int64_t *>(index),
                                    reinterpret_cast<llaisys::bf16_t *>(weight), batch_size, seq_len, vocab_size);
        break;
    default:
        throw std::runtime_error("unsupported dtype");
    }
}

} // namespace llaisys::ops::cpu