#include "../../models/qwen2.hpp"
#include "llaisys_qwen2.hpp"
#include <llaisys/models/qwen2.h>

__C {
    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
        auto model = llaisys::models::createQwen2(meta, device, device_ids, ndevice);
        return new LlaisysQwen2Model{model};
    }
    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
        delete model;
    }
    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model) {
        return model->model->weights.get();
    }

    int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken) {
        return model->model->Infer(token_ids, ntoken);
    }
}