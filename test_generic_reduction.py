import torch
import triton
import triton.language as tl

@triton.jit
def _welford_combine(mean_1, m2_1, weight_1, mean_2, m2_2, weight_2):
    delta = mean_2 - mean_1
    new_weight = weight_1 + weight_2
    w2_over_w = weight_2 / new_weight
    return (
        mean_1 + delta * w2_over_w,
        m2_1 + m2_2 + delta * delta * weight_1 * w2_over_w,
        new_weight,
    )

@triton.jit
def var_mean_kernel(X, out_mean, out_var, BLOCK: tl.constexpr):
    xindex = tl.arange(0, BLOCK)
    x = tl.load(X + xindex)
    mean = x
    m2 = tl.zeros_like(x)
    weight = tl.full(x.shape, 1, x.dtype)
    (mean, m2, weight) = tl.reduce((mean, m2, weight), 0, _welford_combine)
    tl.store(out_mean, mean)
    tl.store(out_var, m2 / weight)

SIZE = 512
device = 'cuda'
dtype = torch.float32
x = torch.rand(SIZE, dtype = dtype, device = device)
out_mean = torch.empty((), dtype = dtype, device = device)
out_var = torch.empty((), dtype = dtype, device = device)

kk=var_mean_kernel[(1, )](x, out_mean, out_var, BLOCK = SIZE)
print(kk.asm['ttgir'])

expect_var, expect_mean = torch.var_mean(x, dim=0, correction=0)
torch.testing.assert_close(out_mean, expect_mean)
torch.testing.assert_close(out_var, expect_var)
