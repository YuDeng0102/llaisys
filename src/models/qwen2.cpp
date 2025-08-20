#include "llaisys/models/qwen2.h"
#include "../llaisys/llaisys_tensor.hpp"
#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rearrange/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "llaisys.h"
#include "llaisys/tensor.h"
#include "qwen2.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>

namespace llaisys::models {
Qwen2_t createQwen2(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
    auto qwen2 = std::make_shared<Qwen2>();
    qwen2->meta = std::make_shared<LlaisysQwen2Meta>(*meta);
    qwen2->weights = std::make_shared<LlaisysQwen2Weights>();
    qwen2->device = device;
    qwen2->device_ids = std::vector<int>(device_ids, device_ids + ndevice);
    std::cerr << "Qwen2 model created with device: " << device << " and device_ids: ";
    for (int i = 0; i < ndevice; ++i) {
        std::cerr << device_ids[i] << " ";
    }
    std::cerr << std::endl;
    // print meta
    std::cerr << "dtype: " << qwen2->meta->dtype << std::endl;
    std::cerr << "nlayer: " << qwen2->meta->nlayer << std::endl;
    std::cerr << "hs: " << qwen2->meta->hs << std::endl;
    std::cerr << "nh: " << qwen2->meta->nh << std::endl;
    std::cerr << "nkvh: " << qwen2->meta->nkvh << std::endl;
    std::cerr << "dh: " << qwen2->meta->dh << std::endl;
    std::cerr << "di: " << qwen2->meta->di << std::endl;
    std::cerr << "maxseq: " << qwen2->meta->maxseq << std::endl;
    std::cerr << "voc: " << qwen2->meta->voc << std::endl;
    std::cerr << "epsilon: " << qwen2->meta->epsilon << std::endl;
    std::cerr << "theta: " << qwen2->meta->theta << std::endl;
    std::cerr << "end_token: " << qwen2->meta->end_token << std::endl;
    std::cerr << "C++ llaisysTensor_t size: " << sizeof(llaisysTensor_t) << std::endl;

    qwen2->weights->attn_norm_w = new llaisysTensor_t[meta->nlayer];
    qwen2->weights->attn_q_w = new llaisysTensor_t[meta->nlayer];
    qwen2->weights->attn_q_b = new llaisysTensor_t[meta->nlayer];
    qwen2->weights->attn_k_w = new llaisysTensor_t[meta->nlayer];
    qwen2->weights->attn_k_b = new llaisysTensor_t[meta->nlayer];
    qwen2->weights->attn_v_w = new llaisysTensor_t[meta->nlayer];
    qwen2->weights->attn_v_b = new llaisysTensor_t[meta->nlayer];
    qwen2->weights->attn_o_w = new llaisysTensor_t[meta->nlayer];
    qwen2->weights->mlp_norm_w = new llaisysTensor_t[meta->nlayer];
    qwen2->weights->mlp_gate_w = new llaisysTensor_t[meta->nlayer];
    qwen2->weights->mlp_down_w = new llaisysTensor_t[meta->nlayer];
    qwen2->weights->mlp_up_w = new llaisysTensor_t[meta->nlayer];

    return qwen2;
}

Qwen2::~Qwen2() {
    delete[] weights->attn_norm_w;
    delete[] weights->attn_q_w;
    delete[] weights->attn_q_b;
    delete[] weights->attn_k_w;
    delete[] weights->attn_k_b;
    delete[] weights->attn_v_w;
    delete[] weights->attn_v_b;
    delete[] weights->attn_o_w;
    delete[] weights->mlp_norm_w;
    delete[] weights->mlp_gate_w;
    delete[] weights->mlp_down_w;
    delete[] weights->mlp_up_w;
}

tensor_t GetSmallPartSlice(tensor_t input, std::vector<size_t> &slice_indices) {
    // Create a new tensor with the same shape and dtype as the input tensor
    tensor_t output = input;
    for (size_t i = 0; i / 2 < std::min(slice_indices.size() / 2, input->shape().size()); i += 2) {
        output = output->slice(i / 2, slice_indices[i], slice_indices[i + 1]);
    }
    return output;
}
int64_t Qwen2::Infer(int64_t *token_ids, size_t ntoken) {
    // embedding
    tensor_t input_ids = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, device, device_ids[0]);
    input_ids->load(token_ids);
    tensor_t input_embedding = Tensor::create({ntoken, meta->hs}, meta->dtype, device, device_ids[0]);
    llaisys::ops::embedding(input_embedding, input_ids, weights->in_embed->tensor);

    // // for debug
    // std::vector<size_t> slice_indices = {
    //     ntoken - 5, ntoken, // slice from ntoken-5 to ntoken
    //     0, 1, 0, 5};

    for (size_t i = 0; i < meta->nlayer; i++) {
        // std::cerr << "layer: " << i << std::endl;

        // GetSmallPartSlice(input_embedding, slice_indices)->debug();
        tensor_t hidden_states = input_embedding->copy();
        // layer-norm

        llaisys::ops::rms_norm(hidden_states, hidden_states, weights->attn_norm_w[i]->tensor, meta->epsilon);
        // 计算q,k,v
        tensor_t q = Tensor::create({ntoken, meta->dh * meta->nh}, meta->dtype, device, device_ids[0]);
        tensor_t k = Tensor::create({ntoken, meta->dh * meta->nkvh}, meta->dtype, device, device_ids[0]);
        tensor_t v = Tensor::create({ntoken, meta->dh * meta->nkvh}, meta->dtype, device, device_ids[0]);
        llaisys::ops::linear(q, hidden_states, weights->attn_q_w[i]->tensor, weights->attn_q_b[i]->tensor);
        llaisys::ops::linear(k, hidden_states, weights->attn_k_w[i]->tensor, weights->attn_k_b[i]->tensor);
        llaisys::ops::linear(v, hidden_states, weights->attn_v_w[i]->tensor, weights->attn_v_b[i]->tensor);

        q = q->view({ntoken, meta->nh, meta->dh});
        k = k->view({ntoken, meta->nkvh, meta->dh});
        v = v->view({ntoken, meta->nkvh, meta->dh});

        // rope
        tensor_t rope_pos = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, device, device_ids[0]);
        std::vector<int64_t> rope_pos_data(ntoken);
        for (size_t j = 0; j < ntoken; j++) {
            rope_pos_data[j] = j;
        }
        rope_pos->load(rope_pos_data.data());
        llaisys::ops::rope(q, q, rope_pos, meta->theta);
        llaisys::ops::rope(k, k, rope_pos, meta->theta);

        // self-attention
        hidden_states = hidden_states->view({ntoken, meta->nh, meta->dh});
        llaisys::ops::self_attention(hidden_states, q, k, v, 1.0f / std::sqrt(meta->dh));
        hidden_states = hidden_states->view({ntoken, meta->hs});
        tensor_t hidden_states_out = Tensor::create({ntoken, meta->hs}, meta->dtype, device, device_ids[0]);
        llaisys::ops::linear(hidden_states_out, hidden_states, weights->attn_o_w[i]->tensor, nullptr);
        llaisys::ops::add(hidden_states, hidden_states_out, input_embedding);

        // layer-norm
        input_embedding = hidden_states->copy();
        llaisys::ops::rms_norm(hidden_states, hidden_states, weights->mlp_norm_w[i]->tensor, meta->epsilon);
        // mlp
        tensor_t mlp_hidden_states = Tensor::create({ntoken, meta->di}, meta->dtype, device, device_ids[0]);
        tensor_t gate_states = Tensor::create({ntoken, meta->di}, meta->dtype, device, device_ids[0]);
        llaisys::ops::linear(mlp_hidden_states, hidden_states, weights->mlp_up_w[i]->tensor, nullptr);
        llaisys::ops::linear(gate_states, hidden_states, weights->mlp_gate_w[i]->tensor, nullptr);

        llaisys::ops::swiglu(mlp_hidden_states, gate_states, mlp_hidden_states);
        llaisys::ops::linear(hidden_states, mlp_hidden_states, weights->mlp_down_w[i]->tensor, nullptr);

        llaisys::ops::add(input_embedding, input_embedding, hidden_states);
    }

    // final layer-norm
    tensor_t last_token = input_embedding->slice(0, ntoken - 1, ntoken);
    llaisys::ops::rms_norm(last_token, last_token, weights->out_norm_w->tensor, meta->epsilon);

    tensor_t logits = Tensor::create({1, meta->voc}, meta->dtype, device, device_ids[0]);
    llaisys::ops::linear(logits, last_token, weights->out_embed->tensor, nullptr);
    logits = logits->view({meta->voc});
    tensor_t max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, device, device_ids[0]);
    tensor_t max_val = Tensor::create({1}, meta->dtype, device, device_ids[0]);
    llaisys::ops::argmax(max_idx, max_val, logits);
    return reinterpret_cast<int64_t *>(max_idx->data())[0];
}
} // namespace llaisys::models