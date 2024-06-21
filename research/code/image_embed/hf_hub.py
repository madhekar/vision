from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="visheratin/MC-LLaVA-3b", filename="configuration_llava.py", local_dir="./", force_download=True)
hf_hub_download(repo_id="visheratin/MC-LLaVA-3b", filename="configuration_phi.py", local_dir="./", force_download=True)
hf_hub_download(repo_id="visheratin/MC-LLaVA-3b", filename="modeling_llava.py", local_dir="./", force_download=True)
hf_hub_download(repo_id="visheratin/MC-LLaVA-3b", filename="modeling_phi.py", local_dir="./", force_download=True)
hf_hub_download(repo_id="visheratin/MC-LLaVA-3b", filename="processing_llava.py", local_dir="./", force_download=True)