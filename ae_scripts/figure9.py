
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import random

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sysname = 'HydraServe'

exec_types = [
    "serverless_vllm",
    "serverlessllm",
    "hydraserve",
    "hydraserve_with_cache"
]

pattern = r"total ttft violation = ([\d\.]+), attainment = ([\d\.]+)"

def fig_cv():
    plt.clf()
    # fig, ax = plt.subplots(figsize=(14, 6.5))

    fig = plt.figure(figsize=(20, 4))
    ax = fig.add_subplot(1, 3, 1)
    
    cvs = [2, 4, 8]
    req = [0.6, 0.7, 0.8]

    # Original experiment results
    naive = [[46.7,  45.7, 43.9], [49.9, 48.6, 44.7], [53.5,  53.3, 48.5]]
    serverlessllm = [[ 57.5, 49.8, 45.3], [61.5, 57.7, 47.7], [60.3, 57.7,  54.0]]
    ours_nocache = [[81.1,  77.2, 73.8], [84.2, 78.2, 71.8], [82.5, 76.3, 75.2]]
    ours = [[83.7, 79.1, 78.3], [88.2,  83.8,  79.3], [88.2, 85.0, 78.5]]
    orig_data = [naive, serverlessllm, ours_nocache, ours]

    analyzed_results = []
    for exec_idx, exec_type in enumerate(exec_types):
        result_exec = []
        for cv_idx, cv in enumerate(cvs):
            results_cv = []
            for req_idx, req_rate in enumerate(req):
                log_path = f"/root/logs/expr_1_{exec_type}_{cv}_{req_rate}.log.res"
                if not os.path.exists(log_path):
                    print(f"Warning: evaluation of exec_type = {exec_type}, cv = {cv}, req_rate = {req_rate} not completed. Use data point from our original experiment.")
                    attainment = orig_data[exec_idx][cv_idx][req_idx]
                else:
                    handler = open(log_path, 'r')
                    lines = handler.readlines()
                    for line in lines:
                        match = re.match(pattern, line)
                        if match:
                            attainment = float(match.group(2))
                results_cv.append(attainment)
            result_exec.append(results_cv)
        analyzed_results.append(result_exec)
    
    naive = analyzed_results[0]
    serverlessllm = analyzed_results[1]
    ours_nocache = analyzed_results[2]
    ours = analyzed_results[3]

    all = [naive[0], serverlessllm[0], ours_nocache[0], ours[0]]

    markers = ['D', 'v', '^', 's', '1', '2', '3', '4']
    linestyles = ['solid', 'solid', 'solid', 'solid', (0,(3,5,1,5)), (5,(10,3))]
    # colors = ['tomato', 'green', 'violet', 'dimgray', 'blue', 'orange']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    labels = ["Serverless vLLM", "ServerlessLLM", sysname, sysname + " with Cache"]
    lw = 2
    ms = 7
    fs = 14
    label_fs = 20

    pos = range(len(req))

    for i in range(0, len(all)):
        rect = plt.plot(pos, all[i], color=colors[i], marker=markers[i], markersize=ms, linestyle=linestyles[i], label=labels[i])

    ax.set_xlabel('Request Rate (req/s)', fontsize=fs)
    # latency
    # ax.set_ylabel('Latency per Token (ms)', fontsize=label_fs)
    # memory
    ax.set_ylabel('SLO Attainment (%)', fontsize=label_fs)
    ax.set_ylim(ymin=40, ymax=100)
    ax.set_title("(a) CV=2", y=-0.3, fontsize=fs)

    plt.yticks(fontsize=fs)
    plt.xticks(pos, req, fontsize=fs)

    bwith = 1
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)

    # plt.legend(fontsize=fs, frameon=False)#, bbox_to_anchor=(1, 1.01))

    ax = fig.add_subplot(1, 3, 2)

    all = [naive[1], serverlessllm[1], ours_nocache[1], ours[1]]

    # for alg in range(4):
    #     all[alg].insert(0, all_pre[alg][1][1])
    #     all[alg].insert(0, all_pre[alg][1][0])

    for i in range(0, len(all)):
        rect = plt.plot(pos, all[i], color=colors[i], marker=markers[i], markersize=ms, linestyle=linestyles[i], label=labels[i])

    ax.set_xlabel('Request Rate (req/s)', fontsize=fs)
    ax.set_ylim(ymin=40, ymax=100)
    ax.set_title("(b) CV=4", y=-0.3, fontsize=fs)

    plt.yticks(fontsize=fs)
    plt.xticks(pos, req, fontsize=fs)

    bwith = 1
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)

    ax = fig.add_subplot(1, 3, 3)

    all = [naive[2], serverlessllm[2], ours_nocache[2], ours[2]]

    # for alg in range(4):
    #     all[alg].insert(0, all_pre[alg][2][1])
    #     all[alg].insert(0, all_pre[alg][2][0])

    for i in range(0, len(all)):
        rect = plt.plot(pos, all[i], color=colors[i], marker=markers[i], markersize=ms, linestyle=linestyles[i], label=labels[i])

    ax.set_xlabel('Request Rate (req/s)', fontsize=fs)
    ax.set_ylim(ymin=40, ymax=100)
    ax.set_title("(c) CV=8", y=-0.3, fontsize=fs)

    plt.yticks(fontsize=fs)
    plt.xticks(pos, req, fontsize=fs)

    bwith = 1
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)


    fig.legend(labels, loc='upper center', ncol=4, fontsize=fs, frameon=False)

    # ax.xaxis.grid(color='gray', linestyle='--', linewidth=5, alpha=0.3)
    # ax.yaxis.grid(color='gray', linestyle='--', linewidth=5, alpha=0.3)

    plt.savefig('figs/figure9.pdf', bbox_inches='tight', pad_inches=0.2)

fig_cv()