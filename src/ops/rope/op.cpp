#include "op.hpp"
#include "cpu/rope.hpp"
#include "llaisys.h"
#include "nvidia/rope.cuh"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_DTYPE(pos_ids->dtype(), LLAISYS_DTYPE_I64);
    if (out->shape().size() != 3 || in->shape().size() != 3 || out->shape()[2] % 2 != 0) {
        throw std::runtime_error("rope shape error!");
    }
    if (!out->isContiguous() || !in->isContiguous() || !pos_ids->isContiguous()) {
        throw std::runtime_error("rope tensor must be contiguous!");
    }

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out, in, pos_ids, theta);
        break;
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rope(out->data(), in->data(), out->dtype(), pos_ids->data(), out->shape()[0], out->shape()[1], out->shape()[2] / 2, theta);
        break;
#endif
    default:
        throw std::runtime_error("rope not support device type!");
    }
}
} // namespace llaisys::ops
