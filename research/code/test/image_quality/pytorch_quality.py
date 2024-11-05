import piq
import torch

# Load images
image1 = torch.randn(1, 3, 250, 253)  # Replace with your actual images
image2 = torch.randn(1, 3, 256, 256)

# Calculate SSIM
ssim_value = piq.ssim(image1, image2, data_range=1.)

# Calculate PSNR
psnr_value = piq.psnr(image1, image2, data_range=1.)

print("SSIM:", ssim_value)
print("PSNR:", psnr_value)