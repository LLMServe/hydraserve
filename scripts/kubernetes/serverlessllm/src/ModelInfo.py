'''
Model Information
Format: model_name: (model_size_in_gb, ttft_slo(ms), tpot_slo(ms), cost_slo(gpu_memory_in_gb*time_in_ms per token), num_replica)
Use should choose between tpot_slo and cost_slo (set one of both to INF)
'''

ModelList = {
    "chatbot": {
        "modelscope/Llama-2-7b-chat-ms": (13, 9000, 200, 1000000, 64),
        "modelscope/Llama-2-13b-chat-ms": (26, 13000, 200, 1000000, 64),
    },
    "code": {
        "modelscope/Llama-2-7b-chat-ms": (13, 9000, 200, 1000000, 64),
        "modelscope/Llama-2-13b-chat-ms": (26, 13000, 200, 1000000, 64),
    },
    "summarization": {
        "modelscope/Llama-2-7b-chat-ms": (13, 9000, 200, 1000000, 64),
        "modelscope/Llama-2-13b-chat-ms": (26, 13000, 200, 1000000, 64),
    },
}

'''
ModelList = {
    "test": {
        "modelscope/Llama-2-7b-chat-ms": (13, 9000, 200, 1000000, 1),
        "modelscope/Llama-2-13b-chat-ms": (26, 13000, 200, 1000000, 1),
        "LLM-Research/Meta-Llama-3-8B-Instruct": (15, 9000, 200, 1000000, 1),
        "facebook/opt-2.7b": (7, 9000, 200, 1000000, 1),
        "facebook/opt-6.7b": (13, 9000, 200, 1000000, 1),
        "facebook/opt-13b": (26, 13000, 200, 1000000, 1),
        "AI-ModelScope/falcon-7b": (13, 9000, 200, 1000000, 1),
        "qwen/Qwen1.5-7B-Chat": (14, 9000, 200, 1000000, 1),
        "qwen/Qwen1.5-14B-Chat": (27, 13000, 200, 1000000, 1),
        "qwen/Qwen1.5-4B-Chat": (8, 9000, 200, 1000000, 1),
        "AI-ModelScope/gemma-7b-it": (16, 9000, 200, 1000000, 1),
        "X-D-Lab/MindChat-Qwen-7B": (14, 9000, 200, 1000000, 1),
        "Xunzillm4cc/Xunzi-Qwen-Chat": (14, 9000, 200, 1000000, 1),
        "AI-ModelScope/CodeLlama-7b-Instruct-hf": (14, 9000, 200, 1000000, 1),
        "FlagAlpha/Llama3-Chinese-8B-Instruct": (15, 9000, 200, 1000000, 1),
        "AI-ModelScope/LLaMA-Pro-8B-Instruct": (15, 9000, 200, 1000000, 1),
        "LLM-Research/Llama3-ChatQA-1.5-8B": (15, 9000, 200, 1000000, 1),
        "AI-ModelScope/codegemma-7b-it": (15, 9000, 200, 1000000, 1),
        "qwen/Qwen2-7B-Instruct": (14, 9000, 200, 1000000, 1),
        "qwen/Qwen-7B-Chat": (14, 9000, 200, 1000000, 1),
        "Shanghai_AI_Laboratory/internlm2-chat-7b": (14, 9000, 200, 1000000, 1),
        "baichuan-inc/Baichuan2-7B-Chat": (14, 9000, 200, 1000000, 1),
        "ZhipuAI/chatglm2-6b": (12, 9000, 200, 1000000, 1),
        "ZhipuAI/chatglm3-6b": (12, 9000, 200, 1000000, 1),
    }
}
'''