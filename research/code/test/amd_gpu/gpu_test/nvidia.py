from transformers import pipeline
import torch


# Initialize pipeline on CPU
# pipe = pipeline('text-generation', model='gpt2', device=-1)

# # Move the model to the first GPU
# if torch.cuda.is_available():
#     pipe.model.to("cuda:0")


model_id = "xtuner/llava-llama-3-8b-v1_1-transformers" #"xtuner/llava-phi-3-mini-hf" #"xtuner/llava-llama-3-8b-hf"
pipe = pipeline("image-to-text", model=model_id, device=-1)

if torch.cuda.is_available():
    pipe.model.to("cuda:0")

print(pipe)