expr=$1
exec_type=$2
model_set=$3
backend=$4
cv=$5
req_rate=$6

cur_dir=$(dirname "$0")

if [ "$expr" == "0" ]; then
    if [ "$backend" == "a10" ]; then
        if [ "$model_set" == "1" ]; then
            echo "No need to run model_set 1 on A10 GPUs. Please move on to next setting."
            exit 1
        fi
        used_machine_types="ecs.gn7i-c32g1.8xlarge"
    else
        used_machine_types="ecs.gn6e-c12g1.12xlarge"
    fi
    export LARGEST=1
else
    used_machine_types="ecs.gn7i-c32g1.32xlarge,ecs.gn6e-c12g1.12xlarge"
    export LARGEST=0
fi

mkdir -p /root/logs
log_path="/root/logs/expr_${expr}_${exec_type}_${model_set}_${backend}_${cv}_${req_rate}_main.log"

echo "Delete existing endpoints..."
kubectl delete deployment --all
ps aux | grep "python src/" | grep -v grep | awk '{print $2}' | xargs kill -9
sleep 3
python $cur_dir/../scripts/kubernetes/vllm/src/clear_shm.py > /dev/zero 2>&1     # clear shared memory created by local storage servers
sleep 3
echo "Existing endpoints deleted."

vllm() {
    cd $cur_dir/../scripts/kubernetes/vllm
    SLOW_EXPR=1 python src/start_storage_server.py
    SLOW_EXPR=1 python src/main.py > $log_path 2>&1 &
    echo "Waiting for endpoint startup..."
    sleep 10
}

serverlessllm() {
    if [ "$expr" == "0" ]; then
        export EXPR_1_1=1
    else
        # Get model list
        python $cur_dir/../scripts/kubernetes/vllm/src/request_generator.py $cur_dir/../scripts/kubernetes/vllm/trace/trace_${cv}.pkl $req_rate model_${cv}_${req_rate}.txt
    fi
    cd $cur_dir/../scripts/kubernetes/serverlessllm
    BACKEND=$backend python src/init_servers.py > $log_path 2>&1 &
    echo "Waiting for endpoint startup..."
    tail -f $log_path | grep -q -- "Uvicorn running on "
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate sllm
    export LLM_SERVER_URL=http://$(cat head_ip.txt):8343/
    MODEL_SET=$model_set BACKEND=$backend python src/deploy_models.py model_${cv}_${req_rate}.txt
}

hydraserve_with_single_worker() {
    cd $cur_dir/../scripts/kubernetes/vllm
    python src/start_storage_server.py
    MAX_PP_SIZE=1 python src/main.py > $log_path 2>&1 &
    echo "Waiting for endpoint startup..."
    sleep 10
}

hydraserve() {
    cd $cur_dir/../scripts/kubernetes/vllm
    python src/start_storage_server.py
    if [ "$expr" == "0" ]; then
        export MAX_PP_SIZE=4
    fi
    python src/main.py > $log_path 2>&1 &
    echo "Waiting for endpoint startup..."
    sleep 10
}

hydraserve_with_cache() {
    cd $cur_dir/../scripts/kubernetes/vllm
    USE_CACHE=1 python src/start_storage_server.py
    USE_CACHE=1 python src/main.py > $log_path 2>&1 &
    echo "Waiting for endpoint startup..."
    sleep 10
}

export MODEL_SET=$model_set
export USED_MACHINES=$used_machine_types
case "$exec_type" in
    "serverless_vllm")
        vllm
        ;;
    "serverlessllm")
        serverlessllm
        ;;
    "serverlessllm_with_cached_model")
        serverlessllm
        ;;
    "hydraserve_with_single_worker")
        hydraserve_with_single_worker
        ;;
    "hydraserve")
        hydraserve
        ;;
    "hydraserve_with_cache")
        hydraserve_with_cache
        ;;
    *)
        echo "unrecognized exec_type: $exec_type"
        exit 1
        ;;
esac

echo "Endpoint started. Log printed to ${log_path}."
