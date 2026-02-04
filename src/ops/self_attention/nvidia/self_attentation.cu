#include "self_attentation.cuh"

template<typename T>
__global__ void compute_qk_kernel(T *attn_score, T *q, T *k, float scale, size_t seq_len, size_t nh, size_t d, size_t total_len, size_t nkvh) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < seq_len * nh * total_len) {
        size_t s = idx / (nh * total_len);
        size_t h = (idx / total_len) % nh;
        size_t t = idx % total_len;
        size_t hkv=h/(nh/nkvh);

        float sum = 0;
        for (size_t i = 0; i < d; i++) {
            T q_val = q[s * nh * d + h * d + i];
            T k_val = k[t * nkvh * d + hkv * d + i];
            sum += to_float(q_val) * to_float(k_val);
        }
        attn_score[idx] = to_cuda_type<T>(sum * scale);
    }
}

template<typename T>
__global__ void compute_softmax_kernel(T *atten_score,size_t seq_len,size_t nh,size_t total_len) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < seq_len * nh ) {
        size_t s = idx / nh;
        size_t h = idx % nh;
  
        // Compute in float precision
        float max_val = to_float(atten_score[s * nh * total_len + h * total_len]);
        size_t last_dim=total_len-seq_len+s;
        for (size_t i = 0; i <=last_dim; i++) {
            float val = to_float(atten_score[s * nh * total_len + h * total_len + i]);
            max_val = fmaxf(max_val, val);
        }
        
        float sum = 0.0f;
        for (size_t i = 0; i <=last_dim; i++) {
            float val = to_float(atten_score[s * nh * total_len + h * total_len + i]);
            sum += expf(val - max_val);
        }
        
        float sum_inv = __frcp_rn(sum);
        for (size_t i = 0; i <total_len; i++) {
            if(i<=last_dim){
                float val = to_float(atten_score[s * nh * total_len + h * total_len + i]);
                atten_score[s * nh * total_len + h * total_len + i] = to_cuda_type<T>(expf(val - max_val) * sum_inv);
            }else {
                atten_score[s * nh * total_len + h * total_len + i] = to_cuda_type<T>(0.0f);
            }
        }
    }
}


template<typename T>
__global__ void compute_attn_val_kernel(T*attn_val,T *attn_score, T *v, size_t seq_len, size_t nh, size_t d, size_t total_len, size_t nkvh) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < seq_len * nh * d) {
        size_t s = idx / (nh * d);
        size_t h = (idx / d) % nh;
        size_t i = idx % d;
        size_t hkv=h/(nh/nkvh);

        float sum = 0;
        for (size_t t = 0; t < total_len; t++) {
            sum += to_float(attn_score[s * nh * total_len + h * total_len + t]) * to_float(v[t * nkvh * d + hkv * d + i]);
        }
        attn_val[idx] = to_cuda_type<T>(sum);
    }


}

template<typename T>
void self_attention_3d_(T *attn_val, T *q, T *k, T *v, float scale,size_t seq_len, size_t nh,
                     size_t d, size_t total_len, size_t nkvh) {
    size_t block_size=256;

    //1. compute q*k^T
    dim3 grid1((seq_len*nh*total_len + block_size - 1) / block_size);
    dim3 block1(block_size);
    T* attn_score;
    CHECK_CUDA(cudaMalloc(&attn_score, sizeof(T)*seq_len*total_len*nh));

    compute_qk_kernel<T><<<grid1, block1>>>(attn_score, q, k, scale, seq_len, nh, d, total_len, nkvh);


    //2. compute softmax
    dim3 grid2((seq_len*nh + block_size - 1) / block_size);
    dim3 block2(block_size);                   
    compute_softmax_kernel<T><<<grid2, block2>>>(attn_score, seq_len, nh, total_len);

    //3. compute attn_score*v
    dim3 grid3((seq_len*nh*d + block_size - 1) / block_size);
    dim3 block3(block_size);
    compute_attn_val_kernel<T><<<grid3, block3>>>(attn_val, attn_score, v, seq_len, nh, d, total_len, nkvh);
    CHECK_CUDA(cudaFree(attn_score));
}

namespace llaisys::ops::nvidia {    
void self_attention_3d(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    switch (attn_val->dtype())
    {
     case LLAISYS_DTYPE_F32:
        self_attention_3d_<float>(reinterpret_cast<float *>(attn_val->data()), reinterpret_cast<float *>(q->data()), reinterpret_cast<float *>(k->data()), reinterpret_cast<float *>(v->data()), scale, attn_val->shape()[0], attn_val->shape()[1], attn_val->shape()[2], k->shape()[0], k->shape()[1]);
        break;
    case LLAISYS_DTYPE_F16:
        self_attention_3d_<__half>(reinterpret_cast<__half *>(attn_val->data()), reinterpret_cast<__half *>(q->data()), reinterpret_cast<__half *>(k->data()), reinterpret_cast<__half *>(v->data()), scale, attn_val->shape()[0], attn_val->shape()[1], attn_val->shape()[2], k->shape()[0], k->shape()[1]);
        break;
    case LLAISYS_DTYPE_BF16:
        self_attention_3d_<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16 *>(attn_val->data()), reinterpret_cast<__nv_bfloat16 *>(q->data()), reinterpret_cast<__nv_bfloat16 *>(k->data()), reinterpret_cast<__nv_bfloat16 *>(v->data()), scale, attn_val->shape()[0], attn_val->shape()[1], attn_val->shape()[2], k->shape()[0], k->shape()[1]);
        break;
    default:
        break;
    }
}
} // namespace llaisys::ops::nvidia