#pragma once
#include "../../../tensor/tensor.hpp"
namespace llaisys::ops::cpu {
void rms_norm_cpu_2d(tensor_t out, tensor_t in, tensor_t weight, float eps);
}