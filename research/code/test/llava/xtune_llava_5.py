from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 1. Load a pre-trained model and tokenizer (e.g., GPT-2)
model_name = "xtuner/llava-llama-3-8b-transformers"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to('cpu')

# 2. Prepare the input prompt
prompt = (
    "Describe the image with thoughtful insights using additional information provided"
)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# 3. Generate text
# max_new_tokens: maximum number of new tokens to generate
# do_sample: set to True for sampling-based generation (e.g., Top-k, Top-p)
# temperature: controls randomness in sampling (higher temperature = more random)
# num_beams: for beam search decoding (set to >1 for beam search)
generated_ids = model.generate(
    input_ids, max_new_tokens=50, do_sample=True, temperature=0.7, top_k=50, top_p=0.95
)

# 4. Decode the output
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print(f"Prompt: {prompt}")
print(f"Generated text: {generated_text}")
