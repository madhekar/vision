
import torch
from piq import ssim, SSIMLoss

x = torch.rand(4, 3, 256, 256, requires_grad=True)
y = torch.rand(4, 3, 256, 256)

ssim_index: torch.Tensor = ssim(x, y, data_range=1.)

loss = SSIMLoss(data_range=1.)
output: torch.Tensor = loss(x, y)
output.backward()