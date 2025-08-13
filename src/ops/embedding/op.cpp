#include "op.hpp"
#include "cpu/embedding_cpu.hpp"
#include "llaisys.h"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    CHECK_SAME_DTYPE(index->dtype(), LLAISYS_DTYPE_I64);

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), out->dtype(), out->shape()[0],
                              out->shape()[1], weight->shape()[0]);
    }
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(), out->dtype(), out->shape()[0],
                              out->shape()[1], weight->shape()[0]);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return embedding_nvidia(out, index, weight);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
