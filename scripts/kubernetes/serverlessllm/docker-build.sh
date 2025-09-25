git clone https://github.com/gjghfd/ServerlessLLM
docker build . -t registry.us-east-1.aliyuncs.com/kubernetes-fc/sllm-serve:v1
docker build -f Dockerfile.worker . -t registry.us-east-1.aliyuncs.com/kubernetes-fc/sllm-serve-worker:v1