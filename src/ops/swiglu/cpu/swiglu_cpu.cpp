#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t numel) {
    for (size_t i = 0; i < numel; i++) {
        // Calculate gate[i] / (1 + exp(-gate[i]))
        // For numerical stability, we use the exp function
        T gate_val = gate[i];
        T sigmoid_val;
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            float gate_f = llaisys::utils::cast<float>(gate_val);
            float sigmoid_f = 1.0f / (1.0f + std::exp(-gate_f));
            sigmoid_val = llaisys::utils::cast<T>(sigmoid_f);
            out[i] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(up[i]) * gate_f * sigmoid_f);
        } else {
            sigmoid_val = T(1.0) / (T(1.0) + std::exp(-gate_val));
            out[i] = up[i] * gate_val * sigmoid_val;
        }
    }
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(gate),
                       reinterpret_cast<const float *>(up), numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(gate),
                       reinterpret_cast<const llaisys::bf16_t *>(up), numel);
    case LLAISYS_DTYPE_F16:
        return swiglu_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(gate),
                       reinterpret_cast<const llaisys::fp16_t *>(up), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu