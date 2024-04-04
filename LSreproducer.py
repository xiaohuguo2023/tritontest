import torch
import triton
import triton.language as tl

@triton.jit()
def kernel(in_out_ptr):
    pid = tl.program_id(axis=0)
    x = tl.load(in_out_ptr + pid)
    out = x * 2
    tl.store(in_out_ptr + pid, out)

for _ in range(2):
    x = torch.ones((65536, ), device='cuda', dtype=torch.float32)
#    if is_hip():
    kernel[(65536, )](x, num_warps=16)  # threads per Warp for ROCM is 64
#    else:
#       kernel[(65536, )](x, num_warps=32)
#    print(x[0])
#    print(pgm.asm['amdgcn'])
    assert torch.all(x == 2)

