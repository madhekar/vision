import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F

# 1. Load a pre-trained model (e.g., ResNet18)
# For a custom model, you would define your model architecture and load its state_dict
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.eval() # Set the model to evaluation mode

# 2. Define image transformations
# These should match the transformations used during training
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 3. Load and preprocess the image
# Replace 'path/to/your/image.jpg' with the actual path to your image
img_path = 'path/to/your/image.jpg'
input_image = Image.open(img_path).convert('RGB')
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # Create a mini-batch as expected by the model

# 4. Make a prediction
with torch.no_grad(): # Disable gradient calculation for inference
    output = model(input_batch)

# 5. Interpret the output
# For ImageNet, there are 1000 classes.
# The output tensor contains raw scores (logits) for each class.
probabilities = F.softmax(output, dim=1) # Convert logits to probabilities
predicted_probability, predicted_class_idx = torch.max(probabilities, 1)

# Optional: Load class labels for better interpretation
# You would need a list or dictionary mapping class indices to names
# For ImageNet, you can get the class labels from a file or a pre-defined list
# Example:
# with open("imagenet_classes.txt", "r") as f:
#     imagenet_classes = [line.strip() for line in f.readlines()]
# predicted_class_name = imagenet_classes[predicted_class_idx.item()]

print(f"Predicted class index: {predicted_class_idx.item()}")
print(f"Predicted probability: {predicted_probability.item():.4f}")
# print(f"Predicted class name: {predicted_class_name}") # If class names are available
