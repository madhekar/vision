from transformers import pipeline
import torch
import os

# Initialize pipeline on CPU
# pipe = pipeline('text-generation', model='gpt2', device=-1)

# # Move the model to the first GPU
# if torch.cuda.is_available():
#     pipe.model.to("cuda:0")

# Source - https://stackoverflow.com/a/74952995
# Posted by chakrr, modified by community. See post 'Timeline' for change history
# Retrieved 2026-01-15, License - CC BY-SA 4.0

torch.cuda.empty_cache()

os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:128"

model_id = "xtuner/llava-llama-3-8b-v1_1-transformers" #"xtuner/llava-phi-3-mini-hf" #"xtuner/llava-llama-3-8b-hf"
pipe = pipeline("image-to-text", model=model_id, device=-1)

if torch.cuda.is_available():
    pipe.model.to("cuda:0")

print(pipe)