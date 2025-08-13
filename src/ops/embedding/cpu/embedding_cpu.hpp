#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void embedding(std::byte *out, std::byte *index, std::byte *weight, llaisysDataType_t dtype, size_t batch_size,
               size_t seq_len, size_t vocab_size);
}
