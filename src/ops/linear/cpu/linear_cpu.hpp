#pragma once
#include "../../../tensor/tensor.hpp"
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {
void linear2d(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias);

}