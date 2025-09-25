import os
import sys
import threading
from ModelInfo import ModelList

if __name__ == '__main__':
    expr_1_1 = True if int(os.getenv("EXPR_1_1", "0")) == 1 else False
    if expr_1_1:
        model_set = int(os.getenv("MODEL_SET", "2"))
        if model_set == 0:
            expr_model_list = [
                "facebook/opt-2.7b",
                "facebook/opt-6.7b",
                "modelscope/Llama-2-7b-chat-ms",
                "LLM-Research/Meta-Llama-3-8B-Instruct",
                "AI-ModelScope/falcon-7b",
            ]
        else:
            expr_model_list = [
                "facebook/opt-13b",
                "modelscope/Llama-2-13b-chat-ms",
            ]
        for model in expr_model_list:
            os.system(f"sllm-cli deploy --model {model}")
        # This model is used to clear cache
        backend = os.getenv("BACKEND", "hybrid")
        if backend == "a10":
            model_name = "modelscope/Llama-2-7b-chat-ms"
        else:
            model_name = "modelscope/Llama-2-13b-chat-ms"
        os.system(f"sllm-cli deploy --model {model_name}")
        print("All models deployed!")
        exit(0)
    
    models = []
    model_path = sys.argv[1]
    with open(model_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            model_name = line
            if model_name.endswith('\n'):
                model_name = model_name[:-1]
                models.append(model_name)

    model_config_path = '/root/model_configs'
    threads = []
    for model in models:
        config_path = os.path.join(model_config_path, model, 'config.json')
        new_thread = threading.Thread(target=os.system, args=(f"sllm-cli deploy --config {config_path}",))
        new_thread.start()
        threads.append(new_thread)
    for thread in threads:
        thread.join()
    print("All models deployed!")