import os

models = [
    "modelscope/Llama-2-7b-chat-ms",
    "modelscope/Llama-2-13b-chat-ms",
    "LLM-Research/Meta-Llama-3-8B-Instruct",
    "AI-ModelScope/falcon-7b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-13b",
]

for model in models:
    os.system(f"sllm-cli deploy --model {model}")