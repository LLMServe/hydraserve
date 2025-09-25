import os
import sys
import time
import asyncio
from kubernetes import client, config
from utils import get_node_list_with_resource

if __name__ == '__main__':
    config.load_kube_config()
    batch_api = client.BatchV1Api()
    core_api = client.CoreV1Api()
    apps_api = client.AppsV1Api()

    node_list = get_node_list_with_resource(core_api)

    for node_name, resource in node_list.items():
        node_mem = resource[3] * 0.9
        server_mem_limit = int(node_mem - (resource[1] + 4) * resource[0])
        mount_cmd = "mount -o size=" + str(server_mem_limit * 1024 * 1024 * 1024) + " -o remount /dev/shm"
        print(mount_cmd)
        cmd = "kubectl node-shell " + node_name + " -- " + mount_cmd
        os.system(cmd)

    print(f"Shared memory limit set!")