from transformers import pipeline
import torch

# Initialize pipeline on CPU
pipe = pipeline('text-generation', model='gpt2', device=-1)

# Move the model to the first GPU
if torch.cuda.is_available():
    pipe.model.to("cuda:0")


print(pipe)