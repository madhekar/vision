import os
import ast
import torch
from torchvision import transforms
from utils.config_util import config

def init():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    (
        filter_model_path,
        data_path,
        filter_model_name,
        filter_model_classes,
        image_size,
        batch_size,
        epocs
    ) = config.filer_config_load()

    batch_size_int = int(batch_size)
    sz = ast.literal_eval(image_size)
    epocs_int = int(epocs)
    image_size_int = (int(sz[0]), int(sz[1]))

    return filter_model_path, data_path, filter_model_name, filter_model_classes, image_size_int, batch_size_int, epocs_int, device

def load_filter_model():

    fmp, dp, fmn, fmc, isz, bz, ei, device = init()

    # 1. Load a pre-trained model (e.g., ResNet18)
    # For a custom model, you would define your model architecture and load its state_dict
    filter_model = torch.load(os.path.join(fmp, fmn), weights_only=False)
    filter_model.to(device)
    class_mapping = torch.load(os.path.join(fmp, fmc))
    class_mapping = {v: k for k, v in class_mapping.items()}
    # class_mapping =  ast.literal_eval(class_mapping)
    print(class_mapping)
    print(filter_model)

    filter_model.eval()  # Set the model to evaluation mode

    filter_model.to(device)

    # 2. Define image transformations
    # These should match the transformations used during training
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(isz),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return filter_model, preprocess, class_mapping, device

fm, pp,cm,device = load_filter_model()

def prep_img_infer(img):

    input_tensor = pp(img)

    input_batch = input_tensor.unsqueeze(0)
    input_batch = input_batch.to(device)

    with torch.no_eval():
        out = fm(input_batch)

    probs = torch.nn.functional.softmax(out[0], dim=0)
    top_prob, top_catid = torch.topk(probs, 1)
    print(f"class {top_catid}, prob: {top_prob.item()}")
    return top_catid

def execute():

    img = "/home/madhekar/work/home-media-app/data/input-data/img/madhekar/2596441a-e02f-588c-8df4-dc66a133fc99/af23fb36-9e89-4b81-9990-020b02fe1056.jpg"

    prep_img_infer(img)