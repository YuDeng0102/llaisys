#include "linear_cpu.hpp"
#include <type_traits>

namespace llaisys::ops::cpu {
template <typename T>
void linear2d_(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    const T *in_ptr = reinterpret_cast<const T *>(in->data());
    const T *weight_ptr = reinterpret_cast<const T *>(weight->data());
    T *out_ptr = reinterpret_cast<T *>(out->data());
    const T *bias_ptr = bias != nullptr ? reinterpret_cast<const T *>(bias->data()) : nullptr;

    size_t batch_size = out->shape()[0];
    size_t in_features = in->shape()[1];
    size_t out_features = out->shape()[1];

    for (size_t b = 0; b < batch_size; b++) {
        for (size_t o = 0; o < out_features; o++) {
            float sum = bias_ptr != nullptr ? llaisys::utils::cast<float>(bias_ptr[o]) : 0;
            float c = 0.0f; // Kahan compensation
            for (size_t i = 0; i < in_features; i++) {
                float product;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    product = llaisys::utils::cast<float>(in_ptr[b * in_features + i]) * llaisys::utils::cast<float>(weight_ptr[o * in_features + i]);
                } else {
                    product = in_ptr[b * in_features + i] * weight_ptr[o * in_features + i];
                }
                float y = product - c;
                float t = sum + y;
                c = (t - sum) - y;
                sum = t;
            }
            out_ptr[b * out_features + o] = llaisys::utils::cast<T>(sum);
        }
    }
}

void linear2d(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    if (out->shape()[0] != in->shape()[0] || out->shape()[1] != weight->shape()[0] || (bias != nullptr && (bias->shape().size() != 1 || bias->shape()[0] != weight->shape()[0]))) {
        throw std::runtime_error("linear op shape error");
    }
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        linear2d_<float>(out, in, weight, bias);
        break;
    case LLAISYS_DTYPE_BF16:
        linear2d_<llaisys::bf16_t>(out, in, weight, bias);
        break;
    case LLAISYS_DTYPE_F16:
        linear2d_<llaisys::fp16_t>(out, in, weight, bias);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
} // namespace llaisys::ops::cpu