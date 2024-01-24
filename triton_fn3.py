import torch
import triton
import triton.language as tl


@triton.jit
def triton_fn(a):
    # Define 'a' as a tensor; you might need to adjust the size and shape as per your requirements
   #` a = torch.tensor([2147483648 - -2147483648], dtype=torch.int64)
    # Convert 'a' to a uint64 tensor
    b = a.to(tl.uint64)  # Adjust the type if necessary

a = torch.tensor([2147 - -21474], dtype=torch.int64)
triton_fn[(1,)](a)

