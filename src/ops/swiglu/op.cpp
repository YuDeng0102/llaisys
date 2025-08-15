#include "op.hpp"

#include "cpu/swiglu_cpu.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());
    if (!out->isContiguous() || !gate->isContiguous() || !up->isContiguous()) {
        throw std::runtime_error("swiglu op only support contiguous tensor");
    }
    // Get the number of elements in the tensor
    size_t numel = out->numel();

    // Get the data pointers
    std::byte *out_data = out->data();
    const std::byte *gate_data = gate->data();
    const std::byte *up_data = up->data();

    // Get the data type
    llaisysDataType_t dtype = out->dtype();

    // Call the CPU implementation
    cpu::swiglu(out_data, gate_data, up_data, dtype, numel);
}
} // namespace llaisys::ops