from huggingface_hub import hf_hub_download
import torch
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor

# Load the model in half-precision
model = LlavaNextVideoForConditionalGeneration.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf", dtype=torch.float16, device_map="auto")
processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")

# Load the video as an np.array, sampling uniformly 8 frames (can sample more for longer videos)
video_path = hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset")

conversation = [
    {

        "role": "user",
        "content": [
            {"type": "text", "text": "Why is this video funny?"},
            {"type": "video", "path": video_path},
            ],
    },
]

inputs = processor.apply_chat_template(conversation, num_frames=8, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")

out = model.generate(**inputs, max_new_tokens=60)
processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)