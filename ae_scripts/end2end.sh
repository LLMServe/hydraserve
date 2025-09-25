exec_type=$1
cv=$2
req_rate=$3

cur_dir=$(dirname "$0")
cd $cur_dir/../scripts/kubernetes/vllm

log_path="/root/logs/expr_1_${exec_type}_${cv}_${req_rate}.log"
main_path="/root/logs/expr_1_${exec_type}_3_hybrid_${cv}_${req_rate}_main.log"

vllm() {
    python src/request_generator.py trace/trace_${cv}.pkl $req_rate > $log_path 2>&1 &
}

serverlessllm() {
    export LLM_SERVER_URL=http://$(cat ../serverlessllm/head_ip.txt):8343/
    SERVERLESS_LLM=1 python src/request_generator.py trace/trace_${cv}.pkl $req_rate > $log_path 2>&1 &
}

case "$exec_type" in
    "serverless_vllm")
        vllm
        ;;
    "serverlessllm")
        serverlessllm
        ;;
    "hydraserve")
        vllm
        ;;
    "hydraserve_with_cache")
        vllm
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

echo "Analyze results."
result_path="${log_path}.res"

if [ "$exec_type" == "serverless_vllm" ] || [ "$exec_type" == "hydraserve" ]; then
    cost_result_path="${result_path}.cost"
    tpot_result_path="${result_path}.tpot"
    COST_LOG=$cost_result_path TPOT_LOG=$tpot_result_path python src/result_analyzer.py $log_path $main_path trace/trace_${cv}.pkl $req_rate > $result_path
else
    python src/result_analyzer.py $log_path > $result_path
fi

echo "Stop all processes..."
kubectl delete deployment --all
ps aux | grep "python src/" | grep -v grep | awk '{print $2}' | xargs kill -9
echo "All processes stopped."