import triton
import triton.language as tl
import torch

ir = f"""
#blocked = #triton_gpu.blocked<{{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}}>
module attributes {{"triton_gpu.compute-capability" = 0 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32}} {{
  tt.func public @var_mean_kernel_0d1d2d(%arg0: !tt.ptr<f32, 1> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<f32, 1> {{tt.divisibility = 16 : i32}}, %arg2: !tt.ptr<f32, 1> {{tt.divisibility = 16 : i32}}) -> () {{
    %cst = arith.constant dense<0.000000e+00> : tensor<512xf32, #blocked>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<512xf32, #blocked>
    %0 = tt.make_range {{end = 512 : i32, start = 0 : i32}} : tensor<512xi32, #blocked>
    %1 = tt.splat %arg0 : (!tt.ptr<f32, 1>) -> tensor<512x!tt.ptr<f32, 1>, #blocked>
    %2 = tt.addptr %1, %0 : tensor<512x!tt.ptr<f32, 1>, #blocked>, tensor<512xi32, #blocked>
    %3 = tt.load %2 {{cache = 1 : i32, evict = 1 : i32, isVolatile = false}} : tensor<512xf32, #blocked>
    %4:3 = "tt.reduce"(%3, %cst, %cst_0) <{{axis = 0 : i32}}> ({{
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32, %arg6: f32, %arg7: f32, %arg8: f32):
      %6 = arith.subf %arg6, %arg3 : f32
      %7 = arith.addf %arg5, %arg8 : f32
      %8 = arith.divf %arg8, %7 : f32
      %9 = arith.mulf %6, %8 : f32
      %10 = arith.addf %arg3, %9 : f32
      %11 = arith.addf %arg4, %arg7 : f32
      %12 = arith.mulf %6, %6 : f32
      %13 = arith.mulf %12, %arg5 : f32
      %14 = arith.mulf %13, %8 : f32
      %15 = arith.addf %11, %14 : f32
      tt.reduce.return %10, %15, %7 : f32, f32, f32
    }}) : (tensor<512xf32, #blocked>, tensor<512xf32, #blocked>, tensor<512xf32, #blocked>) -> (f32, f32, f32)
    tt.store %arg1, %4#0 {{cache = 1 : i32, evict = 1 : i32}} : f32
    %5 = arith.divf %4#1, %4#2 : f32
    tt.store %arg2, %5 {{cache = 1 : i32, evict = 1 : i32}} : f32
    tt.return
  }}
}}
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

kk=var_mean_kernel[(1, 1, 1)](x, out_mean, out_var)
#print(kk.asm['ttgir'])

expect_var, expect_mean = torch.var_mean(x, dim=0, correction=0)
torch.testing.assert_close(out_mean, expect_mean)
torch.testing.assert_close(out_var, expect_var)
