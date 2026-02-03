#pragma once
#include "../../../device/nvidia/common.cuh"
#include "../../../tensor/tensor.hpp"
#include "../../../utils.hpp"
namespace llaisys::ops::nvidia {
void self_attention_3d(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale);
} // namespace llaisys::ops::nvidia