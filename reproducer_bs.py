from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch import device, empty, empty_strided
import triton
import triton.language as tl

# THIS KERNEL IS DEFINED IN TRITON_HELPERS - https://github.com/ROCm/pytorch/blob/rocm6.0_internal_testing/torch/_inductor/triton_helpers.py
# We can potentially add workarounds here if there is no way to solve the issue.
@triton.jit
def bucketize_binary_search(
    values,  # 1D tensor
    offsets_ptr,
    indexing_dtype,
    right,  # bool: if true, use intervals closed on the left; see [Note: Inductor bucketize op]
    OFFSETS_SIZE: int,
    BLOCK_SHAPE,  # tuple/list of block shape
):
    """
    See [Note: Inductor bucketize op]
    """
    low = tl.zeros(BLOCK_SHAPE, dtype=indexing_dtype)
    high = tl.full(BLOCK_SHAPE, OFFSETS_SIZE, dtype=indexing_dtype)
    full_range = tl.full([],OFFSETS_SIZE + 1, dtype=indexing_dtype)
    while full_range > 1:
        mid = (high + low) // 2
        mask = mid < OFFSETS_SIZE
        bucket_upper_bound = tl.load(offsets_ptr + mid, mask=mask)
        if right:
            is_above = values >= bucket_upper_bound
        else:
            is_above = values > bucket_upper_bound
        low = tl.where(is_above & mask, mid + 1, low)
        high = tl.where(is_above, high, mid)
        full_range = (full_range + 1) // 2
    return low

# THIS KERNEL IS AUTOMATICALLY GENERATED BY INDUCTOR
@triton.jit
def triton_fn(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = bucketize_binary_search(tmp0, in_ptr1, tl.int32, True, 10, [XBLOCK])
    tl.store(out_ptr0 + (x0), tmp1, None)


arg0_1 = empty_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.int64)
arg1_1 = empty_strided((10, ), (1, ), device='cuda:0', dtype=torch.int32)
buf0 = empty((64, 64), device='cuda', dtype=torch.int32)
test = triton.compile(triton_fn, signature="*i64,*i32,*i32,*i32", constants={"XBLOCK": 4096})
