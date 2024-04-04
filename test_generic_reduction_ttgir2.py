import triton
import triton.language as tl
import torch

ir = f"""
#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#loc = loc("/home/work/tritontest/./test_generic_reduction.py":17:0)
#loc1 = loc(unknown)
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @var_mean_kernel_0d1d2d(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32} loc("/home/work/tritontest/./test_generic_reduction.py":17:0), %arg1: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32} loc("/home/work/tritontest/./test_generic_reduction.py":17:0), %arg2: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32} loc("/home/work/tritontest/./test_generic_reduction.py":17:0)) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<512xf32, #blocked> loc(#loc1)
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<512xf32, #blocked> loc(#loc1)
    %0 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked1> loc(#loc2)
    %1 = tt.splat %arg0 : !tt.ptr<f32, 1> -> tensor<512x!tt.ptr<f32, 1>, #blocked1> loc(#loc3)
    %2 = tt.addptr %1, %0 : tensor<512x!tt.ptr<f32, 1>, #blocked1>, tensor<512xi32, #blocked1> loc(#loc3)
    %3 = tt.load %2 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<512xf32, #blocked1> loc(#loc4)
    %4 = triton_gpu.convert_layout %3 : tensor<512xf32, #blocked1> -> tensor<512xf32, #blocked> loc(#loc4)
    %5:3 = "tt.reduce"(%4, %cst, %cst_0) <{axis = 0 : i32}> ({
    ^bb0(%arg3: f32 loc(unknown), %arg4: f32 loc(unknown), %arg5: f32 loc(unknown), %arg6: f32 loc(unknown), %arg7: f32 loc(unknown), %arg8: f32 loc(unknown)):
      %7 = arith.subf %arg6, %arg3 : f32 loc(#loc20)
      %8 = arith.addf %arg5, %arg8 : f32 loc(#loc21)
      %9 = arith.divf %arg8, %8 : f32 loc(#loc22)
      %10 = arith.mulf %7, %9 : f32 loc(#loc23)
      %11 = arith.addf %arg3, %10 : f32 loc(#loc24)
      %12 = arith.addf %arg4, %arg7 : f32 loc(#loc25)
      %13 = arith.mulf %7, %7 : f32 loc(#loc26)
      %14 = arith.mulf %13, %arg5 : f32 loc(#loc27)
      %15 = arith.mulf %14, %9 : f32 loc(#loc28)
      %16 = arith.addf %12, %15 : f32 loc(#loc29)
      tt.reduce.return %11, %16, %8 : f32, f32, f32 loc(#loc5)
    }) : (tensor<512xf32, #blocked>, tensor<512xf32, #blocked>, tensor<512xf32, #blocked>) -> (f32, f32, f32) loc(#loc5)
    tt.store %arg1, %5#0 {cache = 1 : i32, evict = 1 : i32} : f32 loc(#loc16)
    %6 = arith.divf %5#1, %5#2 : f32 loc(#loc17)
    tt.store %arg2, %6 {cache = 1 : i32, evict = 1 : i32} : f32 loc(#loc18)
    tt.return loc(#loc19)
  } loc(#loc)
} loc(#loc)
"""

import tempfile
with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
    f.write(ir)
    f.flush()
    var_mean_kernel = triton.compile(f.name)

SIZE = 512
device = 'cuda'
x = torch.rand(SIZE, device=device)
out_mean = torch.empty((), device=device)
out_var = torch.empty((), device=device)

kk=var_mean_kernel[(1, )](x, out_mean, out_var)
print(kk.asm['ttgir'])

expect_var, expect_mean = torch.var_mean(x, dim=0, correction=0)
torch.testing.assert_close(out_mean, expect_mean)
torch.testing.assert_close(out_var, expect_var)
