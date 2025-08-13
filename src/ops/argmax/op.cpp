#include "op.hpp"
#include "cpu/argmax_cpu.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    if (max_idx->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(vals->data(), max_idx->data(), max_val->data(), vals->dtype(), vals->numel());
    }

    llaisys::core::context().setDevice(max_idx->deviceType(), max_idx->deviceId());
    switch (max_idx->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(vals->data(), max_idx->data(), max_val->data(), vals->dtype(), vals->numel());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return argmax_nvidia(max_idx, max_val, vals);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
