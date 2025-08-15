#include "op.hpp"
#include "cpu/rope.hpp"
#include "llaisys.h"

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

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out, in, pos_ids, theta);
    }

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out, in, pos_ids, theta);
        break;
#ifdef LLAISYS_DEVICE_NVIDIA
    case LLAISYS_DEVICE_NVIDIA:
        return device::nvidia::rope(out, in, pos_ids, theta);
        break;
#endif
    default:
        throw std::runtime_error("rope not support device type!");
    }
}
} // namespace llaisys::ops
