from huggingface_hub import hf_hub_download
import torch
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor

'''
Summary for LLaVA-NeXT
LLaVA-NeXT typically requires a specific number of frames sampled uniformly across the video (e.g., 32 or 64). 
OpenCV (cv2) is the most robust and widely used alternative for this purpose if you cannot fix your torchvision installation. 

pip install torch torchvision torchaudio torchcodec av


This
ImportError typically indicates a version mismatch between PyTorch and the NVIDIA NCCL library (or CUDA) in your virtual environment. The symbol ncclDevCommDestroy is part of newer NCCL versions, and older PyTorch builds may fail to find it. 
Here are the solutions, ordered from most likely to least likely:
1. Upgrade/Reinstall PyTorch (Recommended) 
The fastest fix is to reinstall PyTorch, ensuring you have the latest stable version that is compatible with your installed CUDA version.
bash

pip uninstall torch
pip install torch --upgrade --extra-index-url https://download.pytorch.org # Or cu118 for CUDA 11

Based on similar issues, explicitly re-installing helps align the libtorch_cuda.so dependency. 
2. Upgrade the NVIDIA NCCL Library 
If you are using PyTorch 2.1+, it requires newer NCCL versions. You can try installing a newer version of the nvidia-nccl package via pip: 
bash

pip install --upgrade nvidia-nccl

Note: This specific issue often arises when PyTorch tries to use a built-in NCCL that conflicts with system libraries. 
3. Check for Conda/Pip Conflict
If you are using Anaconda, you might have mixed packages from the pytorch channel and conda-forge. The pytorch channel is generally more stable.
bash

conda uninstall pytorch
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia # Or 11.8

Similar issues were solved by removing conda-forge packages and forcing the official pytorch channel. 
4. Set LD_LIBRARY_PATH (If using custom NCCL)
If you are using a system-wide NCCL library, ensure your LD_LIBRARY_PATH includes the correct path to that library. If you are only using pip, this is rarely needed.
Summary of Cause

    Missing Symbol: ncclDevCommDestroy
    Context: libtorch_cuda.so
    Diagnosis: Incompatible PyTorch/NCCL versions. 

If the problem persists, please check your PyTorch CUDA version by running:
python -c "import torch; print(torch.version.cuda)". 


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