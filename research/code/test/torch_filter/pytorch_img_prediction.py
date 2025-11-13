import os
import glob
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F

def load_filter_model():
    # 1. Load a pre-trained model (e.g., ResNet18)
    # For a custom model, you would define your model architecture and load its state_dict
    filter_model = torch.load("filter_model.pth")

    print(filter_model)

    filter_model.eval()# Set the model to evaluation mode

    # 2. Define image transformations
    # These should match the transformations used during training
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return filter_model, preprocess

def predict_type(img_path, class_name, filter_model, preprocess):
    # 3. Load and preprocess the image
    # Replace 'path/to/your/image.jpg' with the actual path to your image
    input_image = Image.open(img_path).convert('RGB')
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # Create a mini-batch as expected by the model

    # 4. Make a prediction
    with torch.no_grad(): # Disable gradient calculation for inference
        output = filter_model(input_batch)

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
    print(f"actuial class: {class_name}")

if __name__=="__main__":
    m, p = load_filter_model()
    testing_root = r"/home/madhekar/temp/filter/testing/"
    try:
        entries =os.listdir(testing_root)
        print(entries)
        for entry in entries:
            print(os.path.join(testing_root, entry))
            loc_path = os.path.join(testing_root, entry)
            print("loc", loc_path)
            if os.path.isdir(loc_path):
                print("--->", entry)
                type = entry
                print()
                files = os.listdir(loc_path)
                print(files)
                for file in files:
                    img_file = os.path.join(loc_path,  file)
                    class_name = type
                    print(img_file, class_name)
                    predict_type(img_file, class_name, m, p)

    except Exception as e:
        print(f"failed with : {e}")                


