import pyiqa
import torch

# List available metrics
print(pyiqa.list_models())

# Create a metric (e.g., LPIPS)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
iqa_metric = pyiqa.create_metric('lpips', device=device)

# Score using image paths
score = iqa_metric('./images/IMG_8645.jpeg', './images/IMG_8646.jpeg')
print("LPIPS score:", score)