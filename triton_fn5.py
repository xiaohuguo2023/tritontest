import torch
from torch import device, empty, empty_strided

import triton
import triton.language as tl
from torch._dynamo.testing import rand_strided

@triton.jit
def randint64(seed, offset, low, high):
    r0, r1, r2, r3 = tl.randint4x(seed, offset)
    r0 = r0.to(tl.uint64)
    r1 = r1.to(tl.uint64)
    result = r0 | (r1 << 32)
    size = high - low
    
    # !!! This line causes the error !!!
    result = result % size.to(tl.uint32)
    
    #result = result.to(tl.int64) + low
    #return result
    return 0

@triton.jit
def triton_fn(in_ptr0, in_ptr1, out_ptr0, load_seed_offset, load_seed_offset1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp12 = tl.load(in_ptr1 + (0))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK])
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = randint64(tmp0, (tmp1).to(tl.uint32), -2147483648, 2147483648)


arg0_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.int32)
buf0 = empty((2, ), device='cuda', dtype=torch.int64)
buf1 = empty((), device='cuda', dtype=torch.int32)

triton_fn[(1,)](buf0, arg0_1, buf1, 0, 1, 1, 1)
