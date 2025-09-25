import os
import sys
import time
import asyncio
from kubernetes import client, config
from model_server_manager import ModelServerManager

if __name__ == '__main__':
    config.load_kube_config()
    core_api = client.CoreV1Api()
    apps_api = client.AppsV1Api()

    print("Start to initialize storage server... It may take several minutes.")

    model_server_manager = ModelServerManager(core_api, apps_api)
    asyncio.run(model_server_manager.init_storage_server())

    print("Storage server initialized!")

