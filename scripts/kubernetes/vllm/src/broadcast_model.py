import os
import re
import sys
import time
import socket
import asyncio
import subprocess
from kubernetes import client, config
from utils import get_node_list_gpu, get_node_list_nogpu

nas_path = os.getenv("MODEL_DIR", "/mnt")

if __name__ == '__main__':
    config.load_kube_config()
    batch_api = client.BatchV1Api()
    core_api = client.CoreV1Api()
    apps_api = client.AppsV1Api()

    broadcast_dir = sys.argv[1]

    src_node = os.getenv("SRC", "MASTER")

    # Get my node name
    my_node_name = None
    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if 'kubelet' in line and '--hostname-override' in line:
            match = re.search(r'--hostname-override=([^\s]+)', line)
            if match:
                my_node_name = match.group(1).strip('"\'')
    if not my_node_name:
        print(f"Error: cannot obtain the node name of master.")
        exit(1)
    if src_node == "SRC":
        gpu_nodes = get_node_list_gpu(core_api)
        src_node_name = gpu_nodes[0]
    else:
        src_node_name = my_node_name

    model_path = os.path.join(nas_path, broadcast_dir)
    zip_name = os.path.join(nas_path, broadcast_dir + ".tar.gz")
    if src_node == "MASTER":
        if os.path.exists(zip_name):
            print(f"Zip file already generated.")
        else:
            print(f"Start to compress files...")
            os.system(f"tar cf - {model_path} | pigz > {zip_name}")
            print(f"End compression.")
    else:
        print(f"Start to compress files...")
        os.system(f"kubectl node-shell {src_node_name} -- sh -c \"[ ! f {zip_name} ] && tar cf - {model_path} | pigz > {zip_name}\"")
        print(f"End compression.")
        print(f"Copy to local directory...")
        os.system(f"kubectl node-shell {src_node_name} -- sh -c \"cat {zip_name}\" > {zip_name}")
        print(f"End Copy")


    node_list = get_node_list_gpu(core_api) + list(get_node_list_nogpu(core_api).keys())

    for node in node_list:
        if node == my_node_name or node == src_node_name:
            continue
        print(f"Copy models to {node}...")
        cmd = f"cat {zip_name} | kubectl node-shell {node} -- sh -c \"cat > {zip_name}\""
        os.system(cmd)
        print(f"Node {node} copied.")
    
    # Decompression
    print(f"Perform decompression...")
    process = []
    for node in node_list:
        if node == my_node_name or node == src_node_name:
            continue
        cmd=f"kubectl node-shell {node} -- sh -c \"mkdir -p {model_path}\""
        os.system(cmd)
        cmd=f"kubectl node-shell {node} -- sh -c \"pigz -dc {zip_name} | tar xf - -C /\""
        p = subprocess.Popen(cmd, shell=True)
        process.append(p)
    
    for p in process:
        p.wait()
    print(f"Model broadcast completed.")
