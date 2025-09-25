exec_type=$1
model_set=$2
backend=$3

cur_dir=$(dirname "$0")
cd $cur_dir/../scripts/kubernetes/vllm

log_path="/root/logs/expr_0_${exec_type}_${model_set}_${backend}.log"

vllm() {
    python src/request_generator.py > $log_path 2>&1 &
}

serverlessllm() {
    export LLM_SERVER_URL=http://$(cat ../serverlessllm/head_ip.txt):8343/
    BACKEND=$backend SERVERLESS_LLM=1 USE_CACHE=0 python src/request_generator.py > $log_path 2>&1 &
}

serverlessllm_with_cached_model() {
    export LLM_SERVER_URL=http://$(cat ../serverlessllm/head_ip.txt):8343/
    BACKEND=$backend SERVERLESS_LLM=1 USE_CACHE=1 python src/request_generator.py > $log_path 2>&1 &
}

hydraserve() {
    python src/request_generator.py > $log_path 2>&1 &
}

export MODEL_SET=$model_set
export EXPR_1_1=1
case "$exec_type" in
    "serverless_vllm")
        vllm
        ;;
    "serverlessllm")
        serverlessllm
        ;;
    "serverlessllm_with_cached_model")
        serverlessllm_with_cached_model
        ;;
    "hydraserve_with_single_worker")
        hydraserve
        ;;
    "hydraserve")
        hydraserve
        ;;
    *)
        echo "unrecognized exec_type: $exec_type"
        exit 1
        ;;
esac

echo "Start to generating requests... Log printed to ${log_path}"
echo "Waiting for measurement completion..."
pid=$!
wait $pid
echo "Experiment done."

echo "Stop all processes..."
kubectl delete deployment --all
ps aux | grep "python src/" | grep -v grep | awk '{print $2}' | xargs kill -9
echo "All processes stopped."