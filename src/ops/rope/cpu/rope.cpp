#include "rope.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace llaisys::ops::cpu {
template <typename T>
void rope_cpu(T *out, const T *in, const int64_t *pos_ids, size_t seq_len, size_t heads, size_t d_model, float theta) {
    auto pih = std::vector(seq_len, std::vector<float>(d_model));
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < d_model; ++j) {
            pih[i][j] = pos_ids[i] / std::pow(theta, static_cast<float>(2 * j) / (d_model * 2));
        }
    }
    auto cos = std::vector(seq_len, std::vector<float>(d_model));
    auto sin = std::vector(seq_len, std::vector<float>(d_model));
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < d_model; ++j) {
            cos[i][j] = std::cos(pih[i][j]);
            sin[i][j] = std::sin(pih[i][j]);
        }
    }
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < heads; ++j) {
            for (size_t k = 0; k < d_model; ++k) {
                // 计算A
                float a0 = llaisys::utils::cast<float>(in[i * heads * d_model * 2 + j * d_model * 2 + k]), b0 = llaisys::utils::cast<float>(in[i * heads * d_model * 2 + j * d_model * 2 + k + d_model]);
                float a1 = a0 * cos[i][k] - b0 * sin[i][k];
                float b1 = b0 * cos[i][k] + a0 * sin[i][k];
                out[i * heads * d_model * 2 + j * d_model * 2 + k] = llaisys::utils::cast<T>(a1);
                out[i * heads * d_model * 2 + j * d_model * 2 + k + d_model] = llaisys::utils::cast<T>(b1);
            }
        }
    }
}

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    size_t seq_len = out->shape()[0], heads = out->shape()[1], d_model = out->shape()[2] / 2;
    int64_t *pos_ids_ptr = reinterpret_cast<int64_t *>(pos_ids->data());
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        rope_cpu(reinterpret_cast<float *>(out->data()), reinterpret_cast<const float *>(in->data()), pos_ids_ptr, seq_len, heads, d_model, theta);
        break;
    case LLAISYS_DTYPE_BF16:
        rope_cpu(reinterpret_cast<llaisys::bf16_t *>(out->data()), reinterpret_cast<const llaisys::bf16_t *>(in->data()), pos_ids_ptr, seq_len, heads, d_model, theta);
        break;
    case LLAISYS_DTYPE_F16:
        rope_cpu(reinterpret_cast<llaisys::fp16_t *>(out->data()), reinterpret_cast<const llaisys::fp16_t *>(in->data()), pos_ids_ptr, seq_len, heads, d_model, theta);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}

} // namespace llaisys::ops::cpu