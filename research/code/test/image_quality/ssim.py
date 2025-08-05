import pyiqa
import torch
from torchvision.transforms import ToTensor
from PIL import Image
import os

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize IQA metrics
# PSNR (Peak Signal-to-Noise Ratio)
psnr_metric = pyiqa.create_metric("psnr", device=device)
# SSIM (Structural Similarity Index)
ssim_metric = pyiqa.create_metric("ssim", device=device)

brisque_metric = pyiqa.create_metric("brisque", device=device)
# Paths to your image directories
ref_dir = "/home/madhekar/work/home-media-app/data/input-data/error/img/quality/AnjaliBackup/20250320-135228/bf98198d-fcc6-51fe-a36a-275c06005669"  # Directory containing original images
dist_dir = "/home/madhekar/work/home-media-app/data/input-data/error/img/quality/AnjaliBackup/20250320-135228/bf98198d-fcc6-51fe-a36a-275c06005669"  # Directory containing distorted images

# Get list of image files (assuming same filenames in both directories)
image_files = [f for f in os.listdir(ref_dir) if f.endswith((".png", ".jpg", ".jpeg"))]

# Loop through images and calculate metrics
for filename in image_files:
    ref_path = os.path.join(ref_dir, filename)
    dist_path = os.path.join(dist_dir, filename)

    # Load images
    ref_img = Image.open(ref_path).convert("RGB")
    dist_img = Image.open(dist_path).convert("RGB")

    # Convert images to PyTorch tensors (N, C, H, W) and move to device
    transform = ToTensor()
    ref_tensor = transform(ref_img).unsqueeze(0).to(device)
    dist_tensor = transform(dist_img).unsqueeze(0).to(device)

    # Calculate metrics
    psnr_score = psnr_metric(ref_tensor, dist_tensor)
    ssim_score = ssim_metric(ref_tensor, dist_tensor)

    brisque_score = brisque_metric(ref_tensor)

    print(
        f"Image: {filename}, PSNR: {psnr_score.item():.2f}, SSIM: {ssim_score.item():.4f} Brinque: {brisque_score.item():.4f}"
    )

# Note: For No-Reference (NR) metrics like BRISQUE, you would only pass the distorted image.
# brisque_metric = pyiqa.create_metric('brisque', device=device)
# brisque_score = brisque_metric(dist_tensor)
# print(f"Image: {filename}, BRISQUE: {brisque_score.item():.2f}")
