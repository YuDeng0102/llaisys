#pragma once
#include "../../../device/nvidia/common.cuh"
#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::nvidia {

void linear2d(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias);

} // namespace llaisys::ops::nvidia
