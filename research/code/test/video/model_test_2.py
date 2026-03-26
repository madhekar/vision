from huggingface_hub import hf_hub_download
import torch
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor

'''
Summary for LLaVA-NeXT
LLaVA-NeXT typically requires a specific number of frames sampled uniformly across the video (e.g., 32 or 64). 
OpenCV (cv2) is the most robust and widely used alternative for this purpose if you cannot fix your torchvision installation. 

pip install torch torchvision torchaudio torchcodec av


'''


# Load the model in half-precision
model = LlavaNextVideoForConditionalGeneration.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf", dtype=torch.float16, device_map="auto")
processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")

# Load the video as an np.array, sampling uniformly 8 frames (can sample more for longer videos)
#video_path = hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="/home/madhekar/Videos/ffmpeg_frames/video_1/VID_20181205_121309.mp4", repo_type="dataset")
custom_path = "/home/madhekar/Videos/ffmpeg_frames/video_1/"
local_video_path = hf_hub_download(repo_id="madhekar/zmedia", filename="VID_20181205_121309.mp4", local_dir=custom_path, local_files_only=True)

conversation = [
    {

        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this video with details"},
            {"type": "video", "path": local_video_path},
            ],
    },
]

inputs = processor.apply_chat_template(conversation, num_frames=8, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")

out = model.generate(**inputs, max_new_tokens=60)
print(out)
processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)