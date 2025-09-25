# HydraServe
HydraServe is a serverless large language model (LLM) serving system designed to minimize cold start latency in public clouds. HydraServe leverages pipeline parallelism to reduce model fetching latency, and adopts pipeline consolidation that merges workers back into individual endpoints to ensure efficient resource usage and high performance for warm requests.
It also speeds up runtime preparation by overlapping distinct stages in the instance startup process, such as model fetching, loading, and library initialization.

It currently supports the following model series:
- OPT
- Llama-2
- Llama-3
- Falcon

## Build && Install

### 1. Prepare Images

We provide the following pre-built images:
```
Our modified vLLM engine: registry.us-east-1.aliyuncs.com/kubernetes-fc/modelscope-vllm:v1

Model downloader and partitioner: chlou/vllm-download:v1

Our model storage server:
- registry.us-east-1.aliyuncs.com/kubernetes-fc/vllm-storage-server:v1
- chlou/vllm-storage-server-remote:v1
```

#### Build Images from Source

1. Build our modified vLLM engine (replace [IMAGE_NAME] with your desired Docker image name)

```
DOCKER_BUILDKIT=1 docker build . --platform=linux/amd64 --tag [IMAGE_NAME] -f scripts/image/Dockerfile
```

2. Build model downloader

```
cd scripts/kubernetes/vllm/download
docker build . --platform=linux/amd64 --tag [IMAGE_NAME]
```

3. Build storage server

```
cd scripts/kubernetes/vllm/storage_server
docker build . --platform=linux/amd64 --tag [IMAGE_NAME_LOCAL_SERVER]
docker build . --platform=linux/amd64 --tag [IMAGE_NAME_REMOTE_SERVER] -f Dockerfile_remote
```

If you build images from source, please configure the image names in [scripts/kubernetes/vllm/src/ImageInfo.py](scripts/kubernetes/vllm/src/ImageInfo.py).

### 2. Setup Environments

If you are using an Aliyun ACK cluster, refer to the environment preparation guide in [Setupenv-aliyun.md](Setupenv-aliyun.md).

1. Environment Requirements

HydraServe needs at least one GPU server to perform inference, and at least one server to act as remote storage.
- Kubernetes v1.30.7+
- Python 3.8+

Log in to a server in your Kubernetes cluster and run the following commands.
```
# The Kubernetes package version must be consistent with the version of your Kubernetes cluster.
pip install kubernetes==30.1.0 modelscope==1.15.0 requests openai fastapi aiohttp uvicorn[standard]
sh scripts/kubernetes/tool-node-shell/setup.sh
```

2. Enable GPU Sharing

Install the [Aliyun GPUShare Plugin](https://github.com/AliyunContainerService/gpushare-scheduler-extender) to enable GPU sharing.

3. Configure Node Labels
   
Label all GPU servers.
```
kubectl label node [node_name] gpu_server=true --overwrite
```

Label all servers with specifications.
Please also configure the specifications of instance types in [scripts/kubernetes/vllm/src/ECSInstance.py](scripts/kubernetes/vllm/src/ECSInstance.py).
```
kubectl label node [node_name] node.kubernetes.io/instance-type=[Instance Type] --overwrite
```

Configure GPU sharing labels.
```
cd scripts/kubernetes
SHARE=1 ALIYUN=0 python label_nodes.py
```

### 3. Fetch Images and Download Models

Run the following commands to fetch the required Docker images and download models. You can obtain your ModelScope token from [ModelScope](https://www.modelscope.cn/my/myaccesstoken).
Models to download are listed in [scripts/kubernetes/vllm/src/ModelInfo.py](scripts/kubernetes/vllm/src/ModelInfo.py), classified in different model sets.
```
cd scripts/kubernetes/vllm
# Fetch images
python src/init_images.py

# Download models from ModelScope
export MODEL_DIR=[PATH_TO_MODEL_DIR]
export MODELSCOPE_TOKEN=[MODELSCOPE_ACCESS_TOKEN]
# If you are using a remote file system shared by all servers, configure the USE_NAS environment variable to 1
export USE_NAS=1
MODEL_SET=2 python src/init_models.py

# Init shared memory limit of GPU servers
python src/init_shm.py              
```

If you only need to download models required in our end-to-end experiment (Section 8.3 in the paper), run
```
MODEL_SET=3 python src/init_models.py
```

If you have downloaded models on the current server, and want to broadcast it to all other servers, run
```
export MODEL_DIR=[PATH_TO_MODEL_DIR]
python src/broadcast_model.py model-cache
```

## Artifact Evaluation

See [ae_scripts/README.md](ae_scripts/README.md) for detailed instructions.

## Code Structure

```
- This repository: Contains our custom serverless LLM serving framework, based on vLLM v0.4.2.
  - scripts/                # Top-level directory for all operational scripts.
    - image/                # Contains Dockerfiles and related assets for building the project's container images.
      - kubernetes/         # All scripts and components for deploying and managing the framework on Kubernetes.
        - vllm/             # The core components of our serverless LLM serving framework.
          - src/            # The main source code for the serving framework logic.
          - download/       # A utility service to download and prepare LLM models, including its Dockerfile.
          - storage_server: # The remote storage server component, including its source code and Dockerfile.
          - trace:          # A tool to generate workload traces for performance end-to-end experiments.
        - serverlessllm:    # Scripts and configurations to run comparative benchmarks against the original ServerlessLLM framework.
        - tool-node-shell:  # Kubernetes node shell installation guide.
```

## Run

### 1. Start HydraServe Endpoint

Start the HydraServe endpoint through Kubernetes.
```
cd scripts/kubernetes/vllm
export USE_CACHE=1 # This command enables local memory cache and is optional
export MODEL_DIR=[PATH_TO_MODEL_DIR]
export LOG_PATH=[PATH_TO_LOG_DIR]           
# Start storage server
python src/start_storage_server.py  
# Start HydraServe endpoint
python src/main.py                  
```

The HydraServe endpoint will be running on `localhost:9090`.

### 2. Testing

1. Send a test chat request to HydraServe
```
curl -X POST -H "Content-Type: application/json" -d '{"id": "0", "model": "modelscope/Llama-2-7b-chat-ms/0", "prompt": "hello"}' http://0.0.0.0:9090
```

2. Generate traces and run an end-to-end experiment

First, follow the instructions in [scripts/kubernetes/vllm/trace/README.md](scripts/kubernetes/vllm/trace/README.md) to generate the workload.

Next, use the request generator to send requests.
```
cd scripts/kubernetes/vllm
python src/request_generator.py [PATH_TO_WORKLOAD] [REQ_PER_SECOND]
```

3. Analyze results
```
cd scripts/kubernetes/vllm
python src/result_analyzer.py [PATH_TO_REQUEST_GENERATOR_OUTPUT_LOG]

# Analyze Cost
python src/result_analyzer.py [PATH_TO_REQUEST_GENERATOR_OUTPUT_LOG] [PATH_TO_HYDRASERVE_OUTPUT_LOG] [PATH_TO_WORKLOAD] [REQ_PER_SECOND]
```

### 3. Stop

To stop storage server, run the following command:
```
python src/close_storage_server.py
```

## Citation

If you use HydraServe in your research, please cite our [paper](https://arxiv.org/abs/2502.15524).

```
@misc{lou2025swiftserverlessllmcold,
      title={Towards Swift Serverless LLM Cold Starts with ParaServe}, 
      author={Chiheng Lou and Sheng Qi and Chao Jin and Dapeng Nie and Haoran Yang and Xuanzhe Liu and Xin Jin},
      year={2025},
      eprint={2502.15524},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2502.15524}, 
}
```