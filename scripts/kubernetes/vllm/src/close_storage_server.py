import os
import sys
import time
import asyncio
from kubernetes import client, config
from model_server_manager import ModelServerManager

from utils import delete_all_deployments

if __name__ == '__main__':
    config.load_kube_config()
    core_api = client.CoreV1Api()
    apps_api = client.AppsV1Api()

    # No need to remove servers
    # model_server_manager = ModelServerManager(core_api, apps_api)
    # try:
    #     asyncio.run(model_server_manager.remove_servers())
    # except Exception as e:
    #     pass

    # delete all existing pods
    delete_all_deployments(core_api, apps_api)

    time.sleep(1)

    os.system("kill -9 $(ps aux | grep \"python\" | awk '{print $2}') ")
