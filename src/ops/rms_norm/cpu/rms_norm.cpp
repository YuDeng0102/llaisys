#include "rms_norm.hpp"
#include <cmath>
#include <cstddef>

namespace llaisys::ops::cpu {
template <typename T>
void rms_norm_cpu_2d_(T *out, T *in, T *weight, float eps, size_t batch_size, size_t hidden_dim) {
    std::vector<float> weights(hidden_dim), input(hidden_dim);
    for (size_t j = 0; j < hidden_dim; j++) {
        weights[j] = llaisys::utils::cast<float>(weight[j]);
    }

    for (size_t i = 0; i < batch_size; i++) {
        float sum = 0;
        for (size_t j = 0; j < hidden_dim; j++) {
            input[j] = llaisys::utils::cast<float>(in[i * hidden_dim + j]);
            sum += input[j] * input[j];
        }
        sum = std::sqrt(sum / hidden_dim + eps);
        for (size_t j = 0; j < hidden_dim; j++) {
            out[i * hidden_dim + j] = llaisys::utils::cast<T>(input[j] * weights[j] / sum);
        }
    }
}

void rms_norm_cpu_2d(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    size_t batch_size = in->shape()[0], hidden_dim = in->shape()[1];
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        rms_norm_cpu_2d_<float>(reinterpret_cast<float *>(out->data()), reinterpret_cast<float *>(in->data()), reinterpret_cast<float *>(weight->data()), eps, batch_size, hidden_dim);
        break;
    case LLAISYS_DTYPE_BF16:
        rms_norm_cpu_2d_<llaisys::bf16_t>(reinterpret_cast<llaisys::bf16_t *>(out->data()), reinterpret_cast<llaisys::bf16_t *>(in->data()), reinterpret_cast<llaisys::bf16_t *>(weight->data()), eps, batch_size, hidden_dim);
        break;
    case LLAISYS_DTYPE_F16:
        rms_norm_cpu_2d_<llaisys::fp16_t>(reinterpret_cast<llaisys::fp16_t *>(out->data()), reinterpret_cast<llaisys::fp16_t *>(in->data()), reinterpret_cast<llaisys::fp16_t *>(weight->data()), eps, batch_size, hidden_dim);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}

} // namespace llaisys::ops::cpu