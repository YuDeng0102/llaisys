#include "op.hpp"
#include "cpu/self_attention.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/self_attentation.cuh"
#endif
namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val, q, k, v, scale);
    }

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val, q, k, v, scale);
        break;
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::self_attention_3d(attn_val, q, k, v, scale);
        break;
#endif
    default:
        TO_BE_IMPLEMENTED();
    }
}
} // namespace llaisys::ops
