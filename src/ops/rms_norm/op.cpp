#include "op.hpp"
#include "cpu/rms_norm.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    if (!out->isContiguous() || !in->isContiguous() || !weight->isContiguous()) {
        throw std::runtime_error("rms_norm op only support contiguous tensor");
    }

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        cpu::rms_norm_cpu_2d(out, in, weight, eps);
    }

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        cpu::rms_norm_cpu_2d(out, in, weight, eps);
        break;
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        cuda::rms_norm_nvidia(out, in, weight, eps);
        break;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
