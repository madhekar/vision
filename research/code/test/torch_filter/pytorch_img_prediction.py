import os
import ast
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
"""

Calculating validation metrics involves assessing the performance of a model or system against a defined set of criteria. 
The specific metrics and calculation methods depend on the nature of the task (e.g., classification, regression, or data validation) and the desired insights.
For Machine Learning Model Validation:
1. Classification Metrics:

    Accuracy: (True Positives + True Negatives) / Total Samples
    Precision: True Positives / (True Positives + False Positives)
    Recall (Sensitivity): True Positives / (True Positives + False Negatives)
    F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
    ROC AUC: Area Under the Receiver Operating Characteristic curve, measuring the trade-off between True Positive Rate and False Positive Rate at various thresholds.
    Confusion Matrix: A table summarizing the counts of True Positives, True Negatives, False Positives, and False Negatives. 

2. Regression Metrics:

    Mean Absolute Error (MAE): Average of the absolute differences between predicted and actual values.
    Mean Squared Error (MSE): Average of the squared differences between predicted and actual values.
    Root Mean Squared Error (RMSE): Square root of the MSE.
    R-squared: Proportion of variance in the dependent variable predictable from the independent variables. 

3. Cross-Validation:

    Divide the dataset into K folds.
    Train the model on K-1 folds and validate on the remaining fold.
    Repeat this process K times, using each fold once for validation.
    Average the metrics across all folds to get a robust estimate of model performance. 

For Data Validation or System Validation:

    Error Rate: Number of failing elements or instances divided by the total number of elements or instances.
    Compliance Rate: Number of compliant elements or instances divided by the total.
    Specific Rule-Based Metrics: Define metrics based on the specific validation rules being applied (e.g., number of missing values, number of values outside a specified range). 

General Steps for Calculation:

    Define the Validation Criteria: Clearly specify what constitutes a "correct" or "valid" outcome.
    Obtain Validation Data: Use a separate dataset (test set) or employ cross-validation techniques to evaluate performance on unseen data.
    Apply the Model/System: Generate predictions or results on the validation data.
    Compare to Ground Truth: Compare the predictions/results to the actual known outcomes or predefined standards.
    Calculate Metrics: Apply the relevant formulas to quantify the performance based on the comparisons.
    Analyze and Interpret: Evaluate the calculated metrics to understand the strengths and weaknesses of the model or system.
"""
def load_filter_model():
    # 1. Load a pre-trained model (e.g., ResNet18)
    # For a custom model, you would define your model architecture and load its state_dict
    filter_model = torch.load("filter_model.pth", weights_only=False)
    class_mapping = torch.load("label_mappings.pth")
    class_mapping =  ast.literal_eval(class_mapping)
    print(class_mapping)
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
    print(f"actual class: {class_name}")

if __name__=="__main__":
    m, p = load_filter_model()
    
    testing_root = "/home/madhekar/temp/filter/testing/"
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


