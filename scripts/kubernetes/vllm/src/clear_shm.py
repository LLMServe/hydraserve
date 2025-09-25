import asyncio
from kubernetes import client, config
from utils import get_node_list_nogpu, get_node_list_gpu

async def clear_shared_memory(node):
    command = f"kubectl node-shell {node} -- find /dev/shm -maxdepth 1 -name '*psm*' -delete"
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    
    if process.returncode == 0:
        print(f"[{node}] Shared memory cleared successfully.")
    else:
        print(f"[{node}] Failed to clear shared memory.\nError: {stderr.decode().strip()}")

async def main():
    config.load_kube_config()
    batch_api = client.BatchV1Api()
    core_api = client.CoreV1Api()
    apps_api = client.AppsV1Api()

    node_list = get_node_list_gpu(core_api) + list(get_node_list_nogpu(core_api).keys())

    print(f"Clearing shared memory...")
    tasks = [clear_shared_memory(node) for node in node_list]
    await asyncio.gather(*tasks)
    print(f"Shared memory cleared.")

if __name__ == '__main__':
    asyncio.run(main())