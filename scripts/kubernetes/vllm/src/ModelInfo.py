'''
Model Information
Format:
- task_type:
    - model_name: (model_size_in_gb, ttft_slo(ms), tpot_slo(ms), cost_slo(gpu_memory_in_gb*time_in_ms per token), num_replica)
'''
import os

ttft_7b = 1500
ttft_13b = 2400
ttft_scale = 5

tpot_7b = 42
tpot_13b = 58
tpot_scale = 2

chat_tpot = 200

scale_ratio = 1

ttft_scale *= scale_ratio
tpot_scale *= scale_ratio
chat_tpot *= scale_ratio

model_set = int(os.getenv("MODEL_SET", "3"))

ModelSet = {}

ModelSet[0] = {
    "test": {
        "facebook/opt-2.7b": (7, 9000, 200, 1000000, 1),
        "facebook/opt-6.7b": (13, 9000, 200, 1000000, 1),
        "AI-ModelScope/falcon-7b": (13, 9000, 200, 1000000, 1),
        "modelscope/Llama-2-7b-chat-ms": (13, 9000, 200, 1000000, 1),
        "LLM-Research/Meta-Llama-3-8B-Instruct": (15, 9000, 200, 1000000, 1),
    }
}

ModelSet[1] = {
    "test": {
        "facebook/opt-13b": (26, 13000, 200, 1000000, 1),
        "modelscope/Llama-2-13b-chat-ms": (26, 13000, 200, 1000000, 1),
    }
}

ModelSet[2] = {
    "test": {
        "modelscope/Llama-2-7b-chat-ms": (13, 9000, 200, 1000000, 1),
        "LLM-Research/Meta-Llama-3-8B-Instruct": (15, 9000, 200, 1000000, 1),
        "facebook/opt-2.7b": (7, 9000, 200, 1000000, 1),
        "facebook/opt-6.7b": (13, 9000, 200, 1000000, 1),
        "AI-ModelScope/falcon-7b": (13, 9000, 200, 1000000, 1),
        "modelscope/Llama-2-13b-chat-ms": (26, 13000, 200, 1000000, 1),
        "facebook/opt-13b": (26, 13000, 200, 1000000, 1),
    }
}

ModelSet[3] = {
    "chatbot": {
        "modelscope/Llama-2-7b-chat-ms": (13, ttft_7b * ttft_scale, chat_tpot, 1000000, 64),
        "modelscope/Llama-2-13b-chat-ms": (25, ttft_13b * ttft_scale, chat_tpot, 1000000, 64),
    },
    "code": {
        "modelscope/Llama-2-7b-chat-ms": (13, ttft_7b * ttft_scale, tpot_7b * tpot_scale, 1000000, 64),
        "modelscope/Llama-2-13b-chat-ms": (25, ttft_13b * ttft_scale, tpot_13b * tpot_scale, 1000000, 64),
    },
    "summarization": {
        "modelscope/Llama-2-7b-chat-ms": (13, ttft_7b * ttft_scale * 5, tpot_7b * tpot_scale, 1000000, 64),
        "modelscope/Llama-2-13b-chat-ms": (25, ttft_13b * ttft_scale * 5, tpot_13b * tpot_scale, 1000000, 64),
    },
}

ModelList = ModelSet[model_set]
