import json
import torch
from PIL import Image
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoProcessor
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

iroot = "/home/madhekar/work/zsource/family/img/"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("microsoft/git-base")   


def loadModel():
    return torch.load("./zgit")

def generate(e):
        # load image
        image = e["image"]
        #width, height = image.size
        #print("=>image", image.resize((int(0.3 * width), int(0.3 * height))))

        # prepare image for the model
        inputs = processor(images=image, return_tensors="pt").to(device)
        pixel_values = inputs.pixel_values

        model = loadModel()
        generated_ids = model.generate(pixel_values=pixel_values, max_length=100)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print("=>", "caption: ", generated_caption)
        
        
if __name__=='__main__':
    
  dataset = load_dataset("imagefolder", data_dir=iroot, split="train", num_proc=8)
  
  print(dataset[0]['image'])
  
  for d in dataset:
      print(d)
      generate(d)
      