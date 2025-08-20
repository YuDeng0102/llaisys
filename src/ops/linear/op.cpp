#include "op.hpp"
#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    if (!out->isContiguous() || !in->isContiguous() || !weight->isContiguous()) {
        throw std::runtime_error("linear op only support contiguous tensor");
    }
    if (bias != nullptr) {
        CHECK_SAME_DEVICE(out, bias);
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
        if (!bias->isContiguous()) {
            throw std::runtime_error("linear op only support contiguous bias");
        }
    }
    // check shape
    if (weight->shape()[1] != *in->shape().rbegin() || weight->shape()[0] != *out->shape().rbegin()) {
        throw std::runtime_error("linear op shape error");
    }
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear2d(out, in, weight, bias);
    }

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear2d(out, in, weight, bias);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return gpu::linear2d(out, in, weight, bias);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
