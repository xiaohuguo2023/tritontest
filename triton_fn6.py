import torch
import triton
import triton.language as tl
 
@triton.jit
def triton_fn():
    a = (2147483648 - -2147483648)
    b = a.to(tl.uint64)

triton_fn[(1,)]()
