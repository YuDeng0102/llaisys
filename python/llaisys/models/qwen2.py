import torch  # Import torch first to avoid CUDA library conflicts
from typing import Sequence
from .. import libllaisys
from ..libllaisys import LIB_LLAISYS,LlaisysQwen2Meta, LlaisysQwen2Weights
from ..libllaisys import DeviceType,llaisysDataTypeStrtoCType, llaisysTensor_t
from ..tensor import Tensor
from pathlib import Path
import safetensors,json
import ctypes
from ctypes import  POINTER,c_int,c_void_p,c_int64
from .utils import *

class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        # 预定义成员变量
        self.device: DeviceType = device
        self.persistent_weights: list = []
        self.dtype = None
        self.nlayer = 0
        self.qwen2 = None
        #初始化权重，防止被gc
        self.w_in_embed = None
        self.w_out_embed = None
        self.w_out_norm = None
        self.w_attn_norm = []
        self.w_attn_q = []
        self.w_attn_q_b = []
        self.w_attn_k = []
        self.w_attn_k_b = []
        self.w_attn_v = []
        self.w_attn_v_b = []
        self.w_attn_o = []
        self.w_mlp_norm = []
        self.w_mlp_gate = []
        self.w_mlp_up = []
        self.w_mlp_down = []
        self.bos_token_id = None
        self.end_token_id = None
        
     
        model_path = Path(model_path)
        weights: dict[str, torch.Tensor] = {}
      
        for file in sorted(model_path.glob("*.safetensors")):
            data_ = safetensors.safe_open(file, framework="pt", device="cpu")
            for name_ in data_.keys():
                weights[name_] = data_.get_tensor(name_)
            
        # 加载config文件
        config = json.load(open(model_path / "config.json"))
        maxseq=1024
        Qwen2Meta = LlaisysQwen2Meta(
            llaisysDataTypeStrtoCType(config["torch_dtype"]),
            config['num_hidden_layers'],
            config['hidden_size'],
            config['num_attention_heads'],
            config['num_key_value_heads'],
            config['hidden_size']//config['num_attention_heads'],
            config['intermediate_size'],
            maxseq,
            config['vocab_size'],
            config['rms_norm_eps'],
            config['rope_theta'],
            config['eos_token_id'],
        )
        self.dtype= Qwen2Meta.dtype
        self.nlayer = Qwen2Meta.nlayer
        self.bos_token_id = config['bos_token_id']
        self.end_token_id = config['eos_token_id']
        ## 调用Qwen2模型创建函数
        device_ids=[0,]
        self.qwen2 = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(Qwen2Meta),
            libllaisys.llaisysDeviceType_t(device),
            (c_int * len(device_ids))(*device_ids),
            len(device_ids),
        )
        print("qwen2 model create",self.dtype, self.nlayer, self.bos_token_id, self.end_token_id)
        self.load_weights(weights)
        
    def load_weights(self, weights_dict: dict[str, torch.Tensor]):
        weights=LIB_LLAISYS.llaisysQwen2ModelWeights(self.qwen2).contents
        # 加载权重
        self.w_in_embed = loadLlaisysTensorWeight(weights_dict, "model.embed_tokens.weight", self.dtype, self.device)
        weights.in_embed = self.w_in_embed.lib_tensor()
        self.w_out_embed = loadLlaisysTensorWeight(weights_dict, "lm_head.weight", self.dtype, self.device)
        weights.out_embed = self.w_out_embed.lib_tensor()
        self.w_out_norm = loadLlaisysTensorWeight(weights_dict, "model.norm.weight", self.dtype, self.device)
        weights.out_norm_w = self.w_out_norm.lib_tensor()
       
        # 加载多层权重
        nlayer = self.nlayer
        
        # 为指针数组分配内存
        # weights.attn_norm_w = (llaisysTensor_t * nlayer)()
        # weights.attn_q_w = (llaisysTensor_t * nlayer)()
        # weights.attn_q_b = (llaisysTensor_t * nlayer)()
        # weights.attn_k_w = (llaisysTensor_t * nlayer)()
        # weights.attn_k_b = (llaisysTensor_t * nlayer)()
        # weights.attn_v_w = (llaisysTensor_t * nlayer)()
        # weights.attn_v_b = (llaisysTensor_t * nlayer)()
        # weights.attn_o_w = (llaisysTensor_t * nlayer)()
        # weights.mlp_norm_w = (llaisysTensor_t * nlayer)()
        # weights.mlp_gate_w = (llaisysTensor_t * nlayer)()
        # weights.mlp_up_w = (llaisysTensor_t * nlayer)()
        # weights.mlp_down_w = (llaisysTensor_t * nlayer)()
        
        # 加载每层的权重
        for i in range(nlayer):
            self.w_attn_norm.append(loadLlaisysTensorWeight(weights_dict, f"model.layers.{i}.input_layernorm.weight", self.dtype, self.device))
            weights.attn_norm_w[i] = self.w_attn_norm[i].lib_tensor()
            
            self.w_attn_q.append(loadLlaisysTensorWeight(weights_dict, f"model.layers.{i}.self_attn.q_proj.weight", self.dtype, self.device))
            weights.attn_q_w[i] = self.w_attn_q[i].lib_tensor()
            
            self.w_attn_q_b.append(loadLlaisysTensorWeight(weights_dict, f"model.layers.{i}.self_attn.q_proj.bias", self.dtype, self.device))
            weights.attn_q_b[i] = self.w_attn_q_b[i].lib_tensor()
            
            self.w_attn_k.append(loadLlaisysTensorWeight(weights_dict, f"model.layers.{i}.self_attn.k_proj.weight", self.dtype, self.device))
            weights.attn_k_w[i] = self.w_attn_k[i].lib_tensor()
            
            self.w_attn_k_b.append(loadLlaisysTensorWeight(weights_dict, f"model.layers.{i}.self_attn.k_proj.bias", self.dtype, self.device))
            weights.attn_k_b[i] = self.w_attn_k_b[i].lib_tensor()
            
            self.w_attn_v.append(loadLlaisysTensorWeight(weights_dict, f"model.layers.{i}.self_attn.v_proj.weight", self.dtype, self.device))
            weights.attn_v_w[i] = self.w_attn_v[i].lib_tensor()
            
            self.w_attn_v_b.append(loadLlaisysTensorWeight(weights_dict, f"model.layers.{i}.self_attn.v_proj.bias", self.dtype, self.device))
            weights.attn_v_b[i] = self.w_attn_v_b[i].lib_tensor()
            
            self.w_attn_o.append(loadLlaisysTensorWeight(weights_dict, f"model.layers.{i}.self_attn.o_proj.weight", self.dtype, self.device))
            weights.attn_o_w[i] = self.w_attn_o[i].lib_tensor()
            
            self.w_mlp_norm.append(loadLlaisysTensorWeight(weights_dict, f"model.layers.{i}.post_attention_layernorm.weight", self.dtype, self.device))
            weights.mlp_norm_w[i] = self.w_mlp_norm[i].lib_tensor()
            
            self.w_mlp_gate.append(loadLlaisysTensorWeight(weights_dict, f"model.layers.{i}.mlp.gate_proj.weight", self.dtype, self.device))
            weights.mlp_gate_w[i] = self.w_mlp_gate[i].lib_tensor()
            
            self.w_mlp_up.append(loadLlaisysTensorWeight(weights_dict, f"model.layers.{i}.mlp.up_proj.weight", self.dtype, self.device))
            weights.mlp_up_w[i] = self.w_mlp_up[i].lib_tensor()
            
            self.w_mlp_down.append(loadLlaisysTensorWeight(weights_dict, f"model.layers.{i}.mlp.down_proj.weight", self.dtype, self.device))
            weights.mlp_down_w[i] = self.w_mlp_down[i].lib_tensor()
            
        
    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int =128,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):

        if not isinstance(inputs, list):
            inputs = list(inputs)
        if not inputs:
            inputs = [self.bos_token_id]
            
        result=inputs.copy()  # 
        # token=self.qwen2.contents.model.meta.bos_token_id
        new_token_len,next_token=0,self.bos_token_id
       
        while new_token_len < max_new_tokens:
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self.qwen2,
                (c_int64 * len(inputs))(*inputs),
                len(inputs),
            )
            result.append(next_token)
            inputs=[next_token]
            new_token_len+=1
            if next_token == self.end_token_id:
                break

        return result