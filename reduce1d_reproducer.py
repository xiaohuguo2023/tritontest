import numpy as np
import triton
import triton.language as tl
import pytest
from somemodule import to_triton, to_numpy, numpy_random, check_type_supported  # Replace 'some_module' with actual module names

dtypes_with_bfloat16 = ['bfloat16']  # Add other data types if needed
num_ctas_list = [1]  # Define your num_ctas_list values
device = 'cuda'  # Assuming CUDA device

@pytest.mark.parametrize("op, dtype_str, shape", 
                         [(op, dtype, shape) for op in ['min'] 
                          for dtype in dtypes_with_bfloat16 
                          for shape in [32, 64, 128, 512]])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_reduce1d(op, dtype_str, shape, num_ctas, device):
    check_type_supported(dtype_str, device)  # Checks if dtype is supported on the device

    # Triton kernel
    @triton.jit
    def kernel(X, Z, BLOCK: tl.constexpr):
        x = tl.load(X + tl.arange(0, BLOCK))
        z = tl.min(x, axis=0)
        tl.store(Z, z)

    # Input
    rs = np.random.RandomState(17)
    x = numpy_random((shape,), dtype_str=dtype_str, rs=rs)

    # NumPy result
    if dtype_str == 'bfloat16':
        z_ref = np.min(x).astype(np.float32)
        z_ref = (z_ref.view('uint32') & np.uint32(0xffff0000)).view('float32')
        z_tri_dtype_str = 'bfloat16'
    else:
        z_ref = np.min(x).astype(dtype_str)

    # Triton result
    x_tri = to_triton(x, device=device)
    if dtype_str == 'bfloat16':
        z_tri = to_triton(np.random.random((1,)).astype(np.float32), device=device, dst_type='bfloat16')
    else:
        z_tri = to_triton(np.random.random((1,)).astype(dtype_str), device=device)

    assert shape is not None and isinstance(shape, int), "shape must be a non-None integer"

    kernel[(1,)](x_tri, z_tri, BLOCK=shape, num_ctas=num_ctas)
    z_tri = to_numpy(z_tri)

    # Compare
    if dtype_str == 'bfloat16':
        np.testing.assert_allclose(z_ref, z_tri, rtol=1e-3, atol=1e-3)
    else:
        np.testing.assert_equal(z_ref, z_tri)

# Example run
test_reduce1d('min', 'int32', 128, 1, 'cuda')

