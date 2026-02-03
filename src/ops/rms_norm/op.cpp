#include "op.hpp"
#include "cpu/rms_norm.hpp"
#include "nvidia/rms_norm.cuh"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    if (!out->isContiguous() || !in->isContiguous() || !weight->isContiguous()) {
        throw std::runtime_error("rms_norm op only support contiguous tensor");
    }

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm_cpu_2d(out, in, weight, eps);
        break;
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rms_norm_2d(out->data(), in->data(), weight->data(), eps, out->dtype(), out->shape()[0], out->shape()[1]);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
