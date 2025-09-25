
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import random

import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sysname = 'HydraServe'

model_to_path = {
    "OPT-2.7B": "facebook/opt-2.7b",
    "OPT-6.7B": "facebook/opt-6.7b",
    "OPT-13B": "facebook/opt-13b",
    "Llama2-7B": "modelscope/Llama-2-7b-chat-ms",
    "Llama2-13B": "modelscope/Llama-2-13b-chat-ms",
    "Llama3-8B": "LLM-Research/Meta-Llama-3-8B-Instruct",
    "Falcon-7B": "AI-ModelScope/falcon-7b"
}

exec_types = [
    "serverless_vllm",
    "serverlessllm",
    "serverlessllm_with_cached_model",
    "hydraserve_with_single_worker",
    "hydraserve"
]

pattern = r"Model (.+): TTFT = ([\d\.]+) s, TPOT = ([\d\.]+) ms"

results = {}

for exec_type in exec_types:
    for backend in ["a10", "v100"]:
        result = {}
        for model_set in ["0", "1"]:
            log_path = f"/root/logs/expr_0_{exec_type}_{model_set}_{backend}.log"
            if os.path.exists(log_path):
                handler = open(log_path, 'r')
                lines = handler.readlines()
                for line in lines:
                    match = re.match(pattern, line)
                    if match:
                        model = match.group(1)
                        ttft = float(match.group(2))
                        result[model] = ttft
        results[(exec_type, backend)] = result

def fig_ttft():
    plt.clf()
    # fig, ax = plt.subplots(figsize=(14, 6.5))

    gs = gridspec.GridSpec(1, 2, width_ratios=[14, 10])
    fig = plt.figure(figsize=(30, 6.5))
    ax = fig.add_subplot(1, 2, 1)

    models = [
        "OPT-2.7B",
        "OPT-6.7B",
        "OPT-13B",
        "Llama2-7B",
        "Llama2-13B",
        "Llama3-8B",
        "Falcon-7B",
    ]

    # Original experiment results
    # naive = [16.78139877319336, 23.625170866648357, 40.26162060101827, 23.16401521364848, 38.57631436983744, 29.212085008621216, 25.427587588628132]
    # serverlessllm = [13.09695315361023, 17.112082481384277, 26.148438692092896, 19.088354349136353, 26.18804693222046, 21.158116102218628, 19.09496283531189]
    # serverlessllm_cache = [11.094237327575684, 11.75343656539917, 14.772999048233032, 12.108455737431845, 15.144047180811564, 13.146969318389893, 12.097854057947794]
    # pp1 = [7.918232282002767, 9.735543807347616, 17.032320658365887, 9.91473356882731, 17.38655996322632, 11.703928788503012, 10.233082214991251]
    # pp4 = [7.930299838383992, 8.184294700622559, 8.524173657099405, 7.655174096425374, 8.706968466440836, 8.565112193425497, 8.297937075297037]
    # all = [naive, serverlessllm, serverlessllm_cache, pp1, pp4]

    all = [[] for _ in range(5)]
    backend = 'v100'
    for idx, exec_type in enumerate(exec_types):
        if (exec_type, backend) not in results:
            print(f"Error: evaluation of exec_type = {exec_type}, backend = {backend} not completed.")
            exit(1)
        result = results[(exec_type, backend)]
        for model in models:
            model_path = model_to_path[model]
            if model_path not in result:
                print(f"Error: evaluation of exec_type = {exec_type}, backend = {backend}, model = {model_path} not completed.")
                exit(1)
            all[idx].append(result[model_path])

    for i in range(len(all)):
        for j in range(len(all[i])):
            all[i][j] = round(all[i][j], 1)

    # for i in range(0, 5):
    #     for j in range(0, 3):
    #         all[i][j] = all[i][j] + 2

    # colors = ['tomato', 'green', 'violet', 'dimgray', 'blue', 'orange']
    colors = ['#A40545', '#F46F44', '#FDD985', '#E9F5A1', '#7FCBA4', '#4B65AF']
    # colors = ['#F27970', '#BB9727', '#54B345', '#32B897', '#05B9E2', '#8983BF']
    labels = ['Serverless vLLM', 'ServerlessLLM', 'ServerlessLLM with cached model', sysname + ' with single worker', sysname]
    lw = 2
    ms = 8
    fs = 17
    label_fs = 20

    w = 0.2
    xpos = [ i*(len(labels)+1)*w for i in range(len(all[0]))]

    for i in range(0, len(all)):
        pos = [t+i*w for t in xpos]
        rect = plt.bar(pos, all[i], width=w,color=colors[i], edgecolor='black',label = labels[i])
        ax.bar_label(rect, padding=3, fontsize=9)

    ax.set_xlabel('(a) Models on V100', fontsize=label_fs)
    # latency
    # ax.set_ylabel('Latency per Token (ms)', fontsize=label_fs)
    # memory
    ax.set_ylabel('TTFT (s)', fontsize=label_fs)
    ax.set_ylim(ymin=0)

    plt.yticks(fontsize=fs)
    plt.xticks([t+1.5*w for t in xpos], models, fontsize=fs)

    bwith = 1
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)

    # plt.legend(fontsize=fs, frameon=False)#, bbox_to_anchor=(1, 1.01))

    ax = fig.add_subplot(1, 2, 2)

    models = [
        "OPT-2.7B",
        "OPT-6.7B",
        "Llama2-7B",
        "Llama3-8B",
        "Falcon-7B",
    ]

    # Original experiment results
    # naive = [10.141360918680826, 17.016832907994587, 16.57095130284627, 19.33425847689311, 16.211599429448444]
    # serverlessllm = [10.045548915863037, 14.098005294799805, 14.077045440673828, 17.072978496551514, 14.104897737503052]
    # serverlessllm_cache = [7.05725359916687, 7.739961306254069, 8.079145113627115, 9.088603417078653, 8.080979108810425]
    # pp1 = [4.6793703238169355, 8.329822301864624, 8.358131885528564, 10.25255831082662, 8.980064471562704]
    # pp4 = [4.7482371239123732, 5.860530376434326, 5.588744242986043, 6.306800206502278, 6.172065019607544]
    # all = [naive, serverlessllm, serverlessllm_cache, pp1, pp4]

    all = [[] for _ in range(5)]
    backend = 'a10'
    for idx, exec_type in enumerate(exec_types):
        if (exec_type, backend) not in results:
            print(f"Error: evaluation of exec_type = {exec_type}, backend = {backend} not completed.")
            exit(1)
        result = results[(exec_type, backend)]
        for model in models:
            model_path = model_to_path[model]
            if model_path not in result:
                print(f"Error: evaluation of exec_type = {exec_type}, backend = {backend}, model = {model_path} not completed.")
                exit(1)
            all[idx].append(result[model_path])

    for i in range(len(all)):
        for j in range(len(all[i])):
            all[i][j] = round(all[i][j], 1)

    w = 0.2
    xpos = [ i*(len(labels)+1)*w for i in range(len(all[0]))]

    for i in range(0, len(all)):
        pos = [t+i*w for t in xpos]
        rect = plt.bar(pos, all[i], width=w,color=colors[i], edgecolor='black',label = labels[i])
        ax.bar_label(rect, padding=3, fontsize=9)

    ax.set_xlabel('(b) Models on A10', fontsize=label_fs)
    # latency
    # ax.set_ylabel('Latency per Token (ms)', fontsize=label_fs)
    # memory
    ax.set_ylabel('TTFT (s)', fontsize=label_fs)
    ax.set_ylim(ymin=0)

    plt.yticks(fontsize=fs)
    plt.xticks([t+1.5*w for t in xpos], models, fontsize=fs)

    bwith = 1
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)

    # plt.legend(fontsize=fs, frameon=False)#, bbox_to_anchor=(1, 1.01))
    fig.legend(labels, loc='upper center', ncol=5, fontsize=fs, frameon=False)

    # ax.xaxis.grid(color='gray', linestyle='--', linewidth=5, alpha=0.3)
    # ax.yaxis.grid(color='gray', linestyle='--', linewidth=5, alpha=0.3)

    plt.savefig('figs/figure7.pdf', bbox_inches='tight', pad_inches=0.2)

fig_ttft()