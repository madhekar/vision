'''
pip install transformers torch Pillow
'''

from transformers import LlavaForConditionalGeneration, LlamaTokenizer
import torch

# Load the model and tokenizer
model_id = "llava-hf/llama3-llava-next-8b-hf"
model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True)
tokenizer = LlamaTokenizer.from_pretrained("llava-hf/llama3-llava-next-8b-hf")

# Example for a text-only input (replace with image loading for actual use)
# A more complex example would involve image pre-processing
prompt = "Hello, how are you?"

# Tokenize the input
# In a real-world scenario, you would load an image and pass it to the tokenizer
# using the multimodal input format.
# Example: inputs = tokenizer.apply_chat_template([{"role": "user", "content": f"<image>\n{prompt}"}], add_generation_prompt=True, return_tensors="pt")
# and then manually add the image to the model inputs.

# For a simple text example:
inputs = tokenizer(prompt, return_tensors="pt").to(model.device, torch.float16)

# Generate text
output_ids = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.1,
        top_p=0.9
    )

# Decode the output
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(generated_text)


