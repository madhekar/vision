import transformers

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition",
    model="microsoft/Phi-3.5-mini-instruct",
    trust_remote_code=True,
)

# pipeline = transformers.pipeline(
#     "text-generation",
#     model="microsoft/phi-4",
#     model_kwargs={"torch_dtype": "auto"},
#     device_map="auto",
# )

# Load model directly
# from transformers import AutoModelForCausalLM

# model = AutoModelForCausalLM.from_pretrained(
#     "microsoft/Phi-4-multimodal-instruct", trust_remote_code=True
# )

messages = [
    {"role": "system", "content": "You are a medieval knight and must provide explanations to modern people."},
    {"role": "user", "content": "How should I explain the Internet?"},
]

outputs = pipeline(messages, max_new_tokens=128)
print(outputs[0]["generated_text"][-1])