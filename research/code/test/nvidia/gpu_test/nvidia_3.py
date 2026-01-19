from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# model = AutoModelForCausalLM.from_pretrained(
#     #"xtuner/llava-llama-3-8b-v1_1-transformers",
#     "facebook/opt-1.3b",
#     quantization_config=bnb_config,
#     device_map="auto" # Automatically map to available devices
# )

pipe_4bit = pipeline(
    model="facebook/opt-1.3b",
     model_kwargs={"quantization_config":bnb_config},
     device_map="auto" # Automatically map to available devices
)

prompt_4bit="4 bit quantized model"

out = pipe_4bit(prompt_4bit, do_sample=True, top_p=0.95, max_new_tokens=50)

print(out[0]['generated_text'])