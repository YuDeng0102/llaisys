#pragma once
#include "llaisys.h"
#include "../../../device/nvidia/common.cuh"

namespace llaisys::ops::nvidia {    
void argmax(std::byte *vals, std::byte *max_idx, std::byte *max_val, llaisysDataType_t dtype, size_t numel);
}
