#pragma once
#include "../../models/qwen2.hpp"
#include "llaisys.h"
__C {
    typedef struct LlaisysQwen2Model {
        llaisys::models::Qwen2_t model;
    } LlaisysQwen2Model;
}