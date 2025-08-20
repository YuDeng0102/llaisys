import ctypes
from ctypes import c_void_p, c_size_t, c_int, c_float, c_int64, Structure, POINTER
from .tensor import llaisysTensor_t
from .llaisys_types import llaisysDataType_t, llaisysDeviceType_t

# Define LlaisysQwen2Meta structure
class LlaisysQwen2Meta(Structure):
    _fields_ = [
        ("dtype", llaisysDataType_t),
        ("nlayer", c_size_t),
        ("hs", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("maxseq", c_size_t),
        ("voc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_int64),
    ]

# Define LlaisysQwen2Weights structure
class LlaisysQwen2Weights(Structure):
    _fields_ = [
        ("in_embed", llaisysTensor_t),
        ("out_embed", llaisysTensor_t),
        ("out_norm_w", llaisysTensor_t),
        ("attn_norm_w", POINTER(llaisysTensor_t)),
        ("attn_q_w", POINTER(llaisysTensor_t)),
        ("attn_q_b", POINTER(llaisysTensor_t)),
        ("attn_k_w", POINTER(llaisysTensor_t)),
        ("attn_k_b", POINTER(llaisysTensor_t)),
        ("attn_v_w", POINTER(llaisysTensor_t)),
        ("attn_v_b", POINTER(llaisysTensor_t)),
        ("attn_o_w", POINTER(llaisysTensor_t)),
        ("mlp_norm_w", POINTER(llaisysTensor_t)),
        ("mlp_gate_w", POINTER(llaisysTensor_t)),
        ("mlp_up_w", POINTER(llaisysTensor_t)),
        ("mlp_down_w", POINTER(llaisysTensor_t)),
    ]

# Define opaque pointer for LlaisysQwen2Model
class LlaisysQwen2Model(ctypes.Structure):
    pass

# Load shared library
def load_qwen2(lib):
    # Define function prototypes
    lib.llaisysQwen2ModelCreate.argtypes = [
        POINTER(LlaisysQwen2Meta),  # meta
        llaisysDeviceType_t,        # device
        POINTER(c_int),             # device_ids
        c_int                       # ndevice
    ]
    lib.llaisysQwen2ModelCreate.restype = POINTER(LlaisysQwen2Model)
    
    lib.llaisysQwen2ModelDestroy.argtypes = [POINTER(LlaisysQwen2Model)]
    lib.llaisysQwen2ModelDestroy.restype = None
    
    lib.llaisysQwen2ModelWeights.argtypes = [POINTER(LlaisysQwen2Model)]
    lib.llaisysQwen2ModelWeights.restype = POINTER(LlaisysQwen2Weights)
    
    lib.llaisysQwen2ModelInfer.argtypes = [
        POINTER(LlaisysQwen2Model),  # model
        POINTER(c_int64),            # token_ids
        c_size_t                     # ntoken
    ]
    lib.llaisysQwen2ModelInfer.restype = c_int64