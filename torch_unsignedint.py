import torch

dtype=torch.uint64
device='cuda'
out_static = torch.zeros((128), dtype=dtype, device=device)
print(out_static)

