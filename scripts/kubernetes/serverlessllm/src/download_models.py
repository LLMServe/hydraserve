import os
from modelscope.hub.api import HubApi
from modelscope.hub.snapshot_download import snapshot_download

models = [
    "modelscope/Llama-2-7b-chat-ms",
    "modelscope/Llama-2-13b-chat-ms",
    "LLM-Research/Meta-Llama-3-8B-Instruct",
    "AI-ModelScope/falcon-7b",
]

token = os.getenv('ACCESS_TOKEN')
HubApi().login(token)

for model_name in models:
    cache_dir = '/mnt/sllm'
    input_dir = snapshot_download(
        model_name,
        cache_dir=cache_dir
    )

huggingface_models = [
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-13b",
]

os.system("mkdir -p /mnt/sllm/facebook")
for model_name in huggingface_models:
    os.system(f"cd /mnt/sllm/facebook && git clone https://huggingface.co/{model_name} && cd {model_name[9:]} && git-lfs pull")