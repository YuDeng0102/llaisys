#include "argmax_cpu.hpp"

#include "../../../utils.hpp"
#include <cmath>

template <typename T>
void argmax_(const T *vals, size_t *max_idx, T *max_val, size_t numel) {
    if (numel == 0) {
        return; // Handle empty case
    }
    max_idx[0] = 0;
    max_val[0] = vals[0];
    float max_val_float = llaisys::utils::cast<float>(max_val[0]);
    for (size_t i = 1; i < numel; i++) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            {
                float val_float = llaisys::utils::cast<float>(vals[i]);
                if (val_float > max_val_float) {
                    max_val_float = val_float;
                    max_idx[0] = i;
                    max_val[0] = vals[i];
                }
            }
        } else {
            if (vals[i] > max_val[0]) {
                max_val[0] = vals[i];
                max_idx[0] = i;
            }
        }
    }
}

namespace llaisys::ops::cpu {
void argmax(std::byte *vals, std::byte *max_idx, std::byte *max_val, llaisysDataType_t dtype, size_t numel) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return argmax_(reinterpret_cast<const float *>(vals), reinterpret_cast<size_t *>(max_idx), reinterpret_cast<float *>(max_val), numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(reinterpret_cast<const llaisys::bf16_t *>(vals), reinterpret_cast<size_t *>(max_idx), reinterpret_cast<llaisys::bf16_t *>(max_val), numel);
    case LLAISYS_DTYPE_F16:
        return argmax_(reinterpret_cast<const llaisys::fp16_t *>(vals), reinterpret_cast<size_t *>(max_idx), reinterpret_cast<llaisys::fp16_t *>(max_val), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu