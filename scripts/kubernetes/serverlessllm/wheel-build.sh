git clone https://github.com/gjghfd/ServerlessLLM
cd ServerlessLLM/serverless_llm/store
docker build -t sllm_store_builder -f Dockerfile.builder .
docker run -it --rm -v $(pwd)/dist:/app/dist sllm_store_builder /bin/bash
export PYTHON_VERSION=310
export TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0"
conda activate py${PYTHON_VERSION} && python setup.py sdist bdist_wheel