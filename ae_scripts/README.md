# Artifact Evaluation

This is the artifact evaluation of NSDI'26 paper "HydraServe: Minimizing Cold Start Latency for Serverless LLM Serving in Public Clouds".
This guide provides instructions to reproduce the main results presented in the paper.

The experiments for this paper were conducted on the [Alibaba Cloud ACK]((https://www.alibabacloud.com/en/product/kubernetes?_p_lc=1)) platform.

## GPU Isolation Configuration
HydraServe and ServerlessLLM require different GPU isolation strategies to function correctly. It is crucial to switch the configuration before running experiments for each system. Note that Serverless vLLM uses the same isolation strategy as HydraServe.

To apply the correct GPU isolation strategy, run the appropriate script to label all GPU servers as follows:

1. For HydraServe and Serverless vLLM experiments:
```
cd scripts/kubernetes
SHARE=1 python label_nodes.py
```
2. For ServerlessLLM experiments:
```
cd scripts/kubernetes
SHARE=0 python label_nodes.py
```

## Figure 7 (Cold Start Latency)

Since the remote storage cannot serve a large number of models concurrently, we have divided the models into two sets. The experiment must be run twice, once for each set.

We provide the script `run_figure7_expr.py` to automate the entire experiment and generate the final figure.
```
cd ae_scripts
python run_figure7_expr.py
```

To manually run each experiment, for each execution type (`serverless_vllm, serverlessllm, serverlessllm_with_cached_model, hydraserve_with_single_worker, hydraserve`), each model set (`0, 1`), and each backend (`a10, v100`), first start the server by
```
export exec_type=[execution_type]
export model_set=[model_set]
export backend=[backend]
sh ./start_server.sh 0 $exec_type $model_set $backend 0 0
```

Then, run the cold start experiment:
```
sh ./coldstart.sh $exec_type $model_set $backend
```

After the experiments for all settings have completed, run `python figure7.py` to generate the figure at `figs/figure7.pdf`.

## Figure 9 (End-to-End Performance)

**NOTE**: You can choose to run a subset of the end-to-end experiments. The plotting script will automatically use our pre-computed results for any settings you do not run. We suggest prioritizing the experiments with `CV=8` and `req_rate=0.6`, as they are required to generate Figures 11 and 13.

For each execution type (`serverless_vllm, serverlessllm, hydraserve, hydraserve_with_cache`), each CV (`8,4,2`), and each request rate (`0.6, 0.7, 0.8`), first start the server by
```
export exec_type=[execution_type]
export cv=[cv]
export req_rate=[request_rate]
sh ./start_server.sh 1 ${exec_type} 3 hybrid ${cv} ${req_rate}
```

Then, run the end-to-end experiment:
```
sh ./end2end.sh $exec_type $cv $req_rate
```
Each experiment will run for approximately one hour.

After the experiments for your selected settings have been completed, run `python figure9.py` to generate the figure `figs/figure9.pdf`.

## Figure 11 (Application Analysis)

This figure requires the results from the end-to-end experiments with `CV=8` and `req_rate=0.6`. Once those are complete, run the following command to generate the figure at `figs/figure11.pdf`:

```
python figure11.py
```

## Figure 13 (TPOT and Cost Penalties)

Similar to Figure 11, this figure requires the results from the end-to-end experiments with `CV=8` and `req_rate=0.6`. Once those are complete, run the following command to generate the figures:

```
python figure13.py
```

This will produce `figs/figure13-a.pdf` and `figs/figure13-b.pdf`.