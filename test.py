import torch
import spconv.pytorch as spconv
from spconv.pytorch import SparseConvTensor

indices = torch.randint(0, 4, (10, 4), device='cuda', dtype=torch.int32)
features = torch.randn(10, 4, device='cuda')
sp = SparseConvTensor(features, indices, spatial_shape=[4,4,4], batch_size=1)

m = spconv.SubMConv3d(4, 8, kernel_size=3, padding=1, indice_key='subm').to('cuda')
out = m(sp)