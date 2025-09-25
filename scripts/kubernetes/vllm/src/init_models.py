import os
import sys
import time
import asyncio
from kubernetes import client, config
from utils import create_job, check_job_fin, delete_job, create_deployment, delete_deployment, get_node_list_gpu, get_node_list_nogpu
from ModelInfo import ModelList 
from ImageInfo import ImageList

nas_path = os.getenv("MODEL_DIR", "/mnt")
modelscope_token = os.getenv("MODELSCOPE_TOKEN", "")
use_nas = True if int(os.getenv("USE_NAS", "0")) == 1 else False

if __name__ == '__main__':
    config.load_kube_config()
    batch_api = client.BatchV1Api()
    core_api = client.CoreV1Api()
    apps_api = client.AppsV1Api()

    print("Start to download models...")

    env = {
        "MODELSCOPE_CACHE": "/mnt/model-cache",
        "MODELSCOPE_TOKEN": modelscope_token,
        "MODELSCOPE_DOMAIN": "www.modelscope.cn",
        "MAX_PP_SIZE": "4",
        "PYTHONUNBUFFERED": "1"
    }

    model_list = ModelList

    node_list = get_node_list_gpu(core_api) + list(get_node_list_nogpu(core_api).keys())

    if use_nas:
        node_list = [node_list[0]]
    
    models_to_download = []
    for task, models in model_list.items():
        for model, model_info in models.items():
            if model not in models_to_download:
                models_to_download.append(model)

    print(f"Download {models_to_download} on {node_list}.")

    index = 0
    for model in models_to_download:
        print(f"Start to download model {model}.")
        env["MODEL_ID"] = model
        if model == "X-D-Lab/MindChat-Qwen-7B":
            env["MODEL_VERSION"] = "v1.0.4"
        else:
            # Use latest revision
            env["MODEL_VERSION"] = "latest"
        name_list = []
        for node in node_list:
            name = "download-" + str(index)
            name_list.append(name)
            index += 1
            create_job(batch_api, name, ImageList["download"], env, node=node, nas_path=nas_path)
        
        while True:
            has_unfin = False
            for name in name_list:
                if not check_job_fin(batch_api, name):
                    has_unfin = True
                    break
            if has_unfin:
                time.sleep(1)
            else:
                for name in name_list:
                    delete_job(batch_api, name)
                break
        print(f"Model {model} initialized.")

    print("All model initialized.")
