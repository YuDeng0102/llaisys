#pragma once
#include "../../../device/nvidia/common.cuh"
#include "../../../tensor/tensor.hpp"
#include "../../../utils.hpp"
namespace llaisys::ops::nvidia {
void rms_norm_2d(std::byte *out, std::byte *in, std::byte *weight, float eps, llaisysDataType_t dtype, size_t dim0,
                 size_t dim1);
}