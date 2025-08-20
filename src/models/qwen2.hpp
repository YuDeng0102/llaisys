#pragma once

#include "../tensor/tensor.hpp"
#include "llaisys.h"
#include "llaisys/models/qwen2.h"
#include "llaisys/tensor.h"
#include <cmath>
#include <memory>
namespace llaisys::models {
struct Qwen2 {
    std::shared_ptr<LlaisysQwen2Meta> meta;
    std::shared_ptr<LlaisysQwen2Weights> weights;
    llaisysDeviceType_t device;
    std::vector<int> device_ids;
    int64_t Infer(int64_t *token_ids, size_t ntoken);
    ~Qwen2();
};
using Qwen2_t = std::shared_ptr<Qwen2>;

Qwen2_t createQwen2(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice);

} // namespace llaisys::models