import torch
from ..tensor import Tensor
from ..libllaisys.llaisys_types import llaisysDataTypeStrtoCType, DeviceType, MemcpyKind, DataType
from ..runtime import RuntimeAPI
# def torchTensor2LlaisysTensor(torch_tensor: torch.Tensor, dtype_name: str, device: DeviceType = DeviceType.CPU, device_id=0)->Tensor:
#     if not torch_tensor.is_contiguous():
#         torch_tensor = torch_tensor.contiguous()
        
#     llaisys_tensor = Tensor(
#         shape=torch_tensor.shape,
#         dtype=llaisysDataTypeStrtoCType(dtype_name),
#         device=device,
#         device_id=device_id,
#     )
#     api = RuntimeAPI(device)
#     bytes_ = torch_tensor.numel() * torch_tensor.element_size()
#     api.memcpy_sync(
#         llaisys_tensor.data_ptr(),
#         torch_tensor.data_ptr(),
#         bytes_,
#         MemcpyKind.D2D,
#     )
    
def torchDevice(device_name: str, device_id=0):
    if device_name == "cpu":
        return torch.device("cpu")
    elif device_name == "nvidia":
        return torch.device(f"cuda:{device_id}")
    else:
        raise ValueError(f"Unsupported device name: {device_name}")
    
    
def loadLlaisysTensorWeight(dict :dict[str,torch.Tensor],name:str,dType:DataType,\
    deviceType:DeviceType,deviceId:int=0)->Tensor:
    if name not in dict:
        raise ValueError(f"Weight {name} not found in dict")
    torch_tensor = dict[name]
    if not torch_tensor.is_contiguous():
        torch_tensor = torch_tensor.contiguous()
    llaisys_tensor = Tensor(
        shape=torch_tensor.shape,
        dtype=dType,
        device=deviceType,
        device_id=deviceId,
    )
    llaisys_tensor.load(torch_tensor.data_ptr())
    return llaisys_tensor
    