import torch
import triton
import triton.language as tl

@triton.jit
def triton_fn():
    # Use Triton's way of creating constants
    a = tl.constexpr(2147483648 - -2147483648)
    b = a.to(uint64)  # In Triton, this might already be a 64-bit integer, depending on how `constexpr` is handled

triton_fn[(1,)]()

