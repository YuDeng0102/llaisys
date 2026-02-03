#pragma once
#include "../../../device/nvidia/common.cuh"
#include "../../../tensor/tensor.hpp"
#include "../../../utils.hpp"

namespace llaisys::ops::nvidia {
void rope(std::byte *out, std::byte *in, llaisysDataType_t dtype, std::byte *pos_ids, size_t seq_len, size_t heads, size_t d_model, float theta);
}