#include "self_attention.hpp"
#include "llaisys.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>

namespace llaisys::ops::cpu {
template <typename T>
void self_attention_(T *attn_val, T *Q, T *K, T *V, float scale, size_t seq_len, size_t nh,
                     size_t d, size_t total_len, size_t nkvh) {
    auto attn_weight = std::vector(seq_len, std::vector(total_len, 0.0f));
    size_t bias = total_len - seq_len;
    for (size_t h = 0; h < nh; h++) {
        size_t hkv = h / (nh / nkvh);
        float sum = 0;
        // size_t max_idx = std::numeric_limits<size_t>::max();
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t j = 0; j <= i + bias; j++) {
                attn_weight[i][j] = 0;
                for (size_t k = 0; k < d; k++) {
                    attn_weight[i][j] += llaisys::utils::cast<float>(Q[i * nh * d + h * d + k]) * llaisys::utils::cast<float>(K[j * nkvh * d + hkv * d + k]) * scale;
                }
            }
            sum = 0;
            float max_val = *std::max_element(attn_weight[i].begin(), attn_weight[i].begin() + i + 1 + bias);
            for (size_t j = 0; j <= i + bias; j++) {
                sum += std::exp(attn_weight[i][j] - max_val);
            }

            for (size_t j = 0; j <= i + bias; j++) {
                attn_weight[i][j] = std::exp(attn_weight[i][j] - max_val) / sum;
            }

            for (size_t k = 0; k < d; k++) {
                float sum = 0;
                for (size_t j = 0; j <= i + bias; j++) {
                    sum += attn_weight[i][j] * llaisys::utils::cast<float>(V[j * nkvh * d + hkv * d + k]);
                }
                attn_val[i * nh * d + h * d + k] = llaisys::utils::cast<T>(sum);
            }
        }
    }
}
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    switch (attn_val->dtype()) {
    case LLAISYS_DTYPE_F32:
        self_attention_<float>(reinterpret_cast<float *>(attn_val->data()),
                               reinterpret_cast<float *>(q->data()),
                               reinterpret_cast<float *>(k->data()),
                               reinterpret_cast<float *>(v->data()),
                               scale,
                               attn_val->shape()[0],
                               attn_val->shape()[1],
                               attn_val->shape()[2],
                               k->shape()[0],
                               k->shape()[1]);
        break;
    case LLAISYS_DTYPE_F16:
        self_attention_<llaisys::fp16_t>(reinterpret_cast<llaisys::fp16_t *>(attn_val->data()),
                                         reinterpret_cast<llaisys::fp16_t *>(q->data()),
                                         reinterpret_cast<llaisys::fp16_t *>(k->data()),
                                         reinterpret_cast<llaisys::fp16_t *>(v->data()),
                                         scale,
                                         attn_val->shape()[0],
                                         attn_val->shape()[1],
                                         attn_val->shape()[2],
                                         k->shape()[0],
                                         k->shape()[1]);
        break;
    case LLAISYS_DTYPE_BF16:
        self_attention_<llaisys::bf16_t>(reinterpret_cast<llaisys::bf16_t *>(attn_val->data()),
                                         reinterpret_cast<llaisys::bf16_t *>(q->data()),
                                         reinterpret_cast<llaisys::bf16_t *>(k->data()),
                                         reinterpret_cast<llaisys::bf16_t *>(v->data()),
                                         scale,
                                         attn_val->shape()[0],
                                         attn_val->shape()[1],
                                         attn_val->shape()[2],
                                         k->shape()[0],
                                         k->shape()[1]);
        break;
    default:
        break;
    }
}
} // namespace llaisys::ops::cpu