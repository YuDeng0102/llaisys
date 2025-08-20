import ctypes
from enum import IntEnum


# Device Type enum
class DeviceType(IntEnum):
    CPU = 0
    NVIDIA = 1
    COUNT = 2


llaisysDeviceType_t = ctypes.c_int


# Data Type enum
class DataType(IntEnum):
    INVALID = 0
    BYTE = 1
    BOOL = 2
    I8 = 3
    I16 = 4
    I32 = 5
    I64 = 6
    U8 = 7
    U16 = 8
    U32 = 9
    U64 = 10
    F8 = 11
    F16 = 12
    F32 = 13
    F64 = 14
    C16 = 15
    C32 = 16
    C64 = 17
    C128 = 18
    BF16 = 19


llaisysDataType_t = ctypes.c_int

# Memory Copy Kind enum
class MemcpyKind(IntEnum):
    H2H = 0
    H2D = 1
    D2H = 2
    D2D = 3


llaisysMemcpyKind_t = ctypes.c_int

# Stream type (opaque pointer)
llaisysStream_t = ctypes.c_void_p


def llaisysDataTypeStrtoCType(data_type: str):
    if data_type == "bfloat16":
        return DataType.BF16
    elif data_type == "float16":
        return DataType.F16
    elif data_type == "float32":
        return DataType.F32
    elif data_type == "int8":
        return DataType.I8
    elif data_type == "int16":
        return DataType.I16
    elif data_type == "int32":
        return DataType.I32
    elif data_type == "int64":
        return DataType.I64
    elif data_type == "uint8":
        return DataType.U8
    elif data_type == "uint16":
        return DataType.U16
    elif data_type == "uint32":
        return DataType.U32
    elif data_type == "uint64":
        return DataType.U64
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    
    
__all__ = [
    "llaisysDeviceType_t",
    "DeviceType",
    "llaisysDataType_t",
    "DataType",
    "llaisysMemcpyKind_t",
    "MemcpyKind",
    "llaisysStream_t",
    "llaisysDataTypeStrtoCType"
]
