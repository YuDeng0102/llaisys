#include "tensor.hpp"

#include "../utils.hpp"
#include "llaisys.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type, int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() { return _storage->memory() + _offset; }

const std::byte *Tensor::data() const { return _storage->memory() + _offset; }

size_t Tensor::ndim() const { return _meta.shape.size(); }

const std::vector<size_t> &Tensor::shape() const { return _meta.shape; }

const std::vector<ptrdiff_t> &Tensor::strides() const { return _meta.strides; }

llaisysDataType_t Tensor::dtype() const { return _meta.dtype; }

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const { return _storage->deviceId(); }

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1),
                           std::multiplies<size_t>());
}

size_t Tensor::elementSize() const { return utils::dsize(_meta.dtype); }

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape,
                const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape,
                 const std::vector<ptrdiff_t> &strides,
                 llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides,
                          0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides,
                          0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides,
                          0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides,
                          0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides,
                          0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides,
                          0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides,
                          0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides,
                          0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides,
                          0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides,
                          0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides,
                          0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(), this->data(), this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(),
                    this->dtype());
    }
}

bool Tensor::isContiguous() const {
    const auto &shape = _meta.shape;
    const auto &stride = _meta.strides;

    if (shape.empty()) {
        return true; // 0维张量默认为连续
    }
    if (shape.size() == 1) {
        return true; // 1维张量默认为连续
    }

    // 检查最后一个维度的步长是否为1
    if (stride.back() != 1) {
        return false;
    }

    // 从倒数第二个维度向前检查步长是否符合行优先规则
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        if (static_cast<size_t>(stride[i]) != stride[i + 1] * shape[i + 1]) {
            return false;
        }
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    if (order.size() != _meta.shape.size()) {
        throw std::runtime_error(
            "Permute order size must be equal to tensor dimension size.");
    }

    // 检查order是否包含所有维度
    std::vector<bool> visited(order.size(), false);
    for (size_t dim : order) {
        if (dim >= _meta.shape.size()) {
            throw std::runtime_error(
                "Permute order contains invalid dimension index.");
        }
        visited[dim] = true;
    }
    for (bool v : visited) {
        if (!v) {
            throw std::runtime_error(
                "Permute order must contain all tensor dimensions.");
        }
    }

    // 计算新的形状和步长
    std::vector<size_t> new_shape(order.size());
    std::vector<ptrdiff_t> new_strides(order.size());
    for (size_t i = 0; i < order.size(); ++i) {
        new_shape[i] = _meta.shape[order[i]];
        new_strides[i] = _meta.strides[order[i]];
    }
    TensorMeta new_meta{_meta.dtype, new_shape, new_strides};

    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage));
}

// 替换原有的view函数实现
tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    // 计算原张量的总元素数
    size_t original_total = numel();

    // 计算新形状的总元素数
    size_t new_total = 1;
    for (size_t dim : shape) {
        new_total *= dim;
    }

    // 检查元素总数是否匹配
    if (original_total != new_total) {
        throw std::runtime_error(
            "Shape incompatible: total number of elements must match. "
            "Original: "
            + std::to_string(original_total) + ", New: " + std::to_string(new_total));
    }

    // 检查是否为连续张量
    if (!isContiguous()) {
        throw std::runtime_error(
            "Cannot view non-contiguous tensor. Use reshape() instead.");
    }

    // 计算新的步长（行优先格式）
    std::vector<ptrdiff_t> new_strides(shape.size());
    if (!shape.empty()) {
        new_strides.back() = 1;
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
            new_strides[i] = new_strides[i + 1] * shape[i + 1];
        }
    }

    // 创建新的TensorMeta
    TensorMeta new_meta{_meta.dtype, shape, new_strides};

    // 创建新张量，共享相同的存储和偏移
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    if (dim >= _meta.shape.size()) {
        throw std::runtime_error("Slice dimension out of range.");
    }
    if (start >= end || end > _meta.shape[dim]) {
        throw std::runtime_error("Invalid slice range.");
    }
    if (start == 0 && end == _meta.shape[dim]) {
        // 如果切片范围是整个维度，则返回原始张量
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    }
    // 计算新的形状和步长
    std::vector<size_t> new_shape = _meta.shape;
    new_shape[dim] = end - start;
    std::vector<ptrdiff_t> new_strides = _meta.strides;

    // 计算新的偏移量
    size_t new_offset = _offset + _meta.strides[dim] * start * elementSize();

    TensorMeta new_meta{_meta.dtype, new_shape, new_strides};

    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, new_offset));
}

// 目前只实现了最高维度上的，用于kv-cache拼接
tensor_t Tensor::cat(const tensor_t &other, size_t dim = 0, tensor_t out = nullptr) const {
    if (dim >= _meta.shape.size() || dim >= other->_meta.shape.size()) {
        throw std::runtime_error("Dimension out of range for concatenation.");
    }

    // 检查其他维度是否匹配
    for (size_t i = 0; i < _meta.shape.size(); ++i) {
        if (i != dim && _meta.shape[i] != other->_meta.shape[i]) {
            throw std::runtime_error("Shapes do not match for concatenation.");
        }
    }

    // 计算新的形状和步长
    std::vector<size_t> new_shape = _meta.shape;
    new_shape[dim] += other->_meta.shape[dim];
    std::vector<ptrdiff_t> new_strides = _meta.strides;

    tensor_t new_tensor = out == nullptr ? create(new_shape, this->dtype(), this->deviceType(), this->deviceId()) : out;
    core::context().runtime().api()->memcpy_sync(
        new_tensor->data(), this->data(), this->numel() * this->elementSize(),
        LLAISYS_MEMCPY_D2D);
    size_t offset = this->numel() * this->elementSize();
    core::context().runtime().api()->memcpy_sync(
        new_tensor->data() + offset, other->data(), other->numel() * other->elementSize(),
        LLAISYS_MEMCPY_D2D);
    return new_tensor;
}

void Tensor::load(const void *src_) {
    // TO_BE_IMPLEMENTED();
    auto src = static_cast<const std::byte *>(src_);
    core::context().runtime().api()->memcpy_sync(
        this->data(), src, this->numel() * this->elementSize(),
        LLAISYS_MEMCPY_H2D);
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    tensor_t res = create(_meta.shape, _meta.dtype, device_type, device);
    if(device_type==deviceType()){
        core::context().runtime().api()->memcpy_sync(
            res->data(), data(), numel() * elementSize(), LLAISYS_MEMCPY_D2D);
    } else {
        if(device_type==LLAISYS_DEVICE_CPU){
            core::context().runtime().api()->memcpy_sync(
                res->data(), data(), numel() * elementSize(), LLAISYS_MEMCPY_D2H);
        } else if(this->deviceType()==LLAISYS_DEVICE_CPU){
            core::context().setDevice(device_type, device);
            core::context().runtime().api()->memcpy_sync(
                res->data(), data(), numel() * elementSize(), LLAISYS_MEMCPY_H2D);
        } 
    }
    return res;
}

tensor_t Tensor::copy() const {
    // 必须连续内存
    tensor_t res = create(_meta.shape, _meta.dtype, deviceType(), deviceId());
    core::context().runtime().api()->memcpy_sync(
        res->data(), data(), numel() * elementSize(), LLAISYS_MEMCPY_D2D);
    return res;
}

} // namespace llaisys