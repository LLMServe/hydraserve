
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

pattern = r"Task (.+) ttft vio = ([\d\.]+), tpot vio = ([\d\.]+), ttft attain = ([\d\.]+), tpot attain = ([\d\.]+)"

def fig_app():
    plt.clf()
    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    
    app = ['Chatbot', 'Code', 'Summarization']
    app_real = ['chatbot', 'code', 'summarization']

    # Original experiment results
    # naive = [51.8, 45.0, 91.5]
    # serverlessllm = [60.6, 48.1, 99.6]
    # ours_nocache = [83.5, 76.6, 97.3]
    # ours = [89.5, 83.0, 99.1]
    # all = [naive, serverlessllm, ours_nocache, ours]

    all = []
    for exec_type in exec_types:
        result_app = []
        log_path = f"/root/logs/expr_1_{exec_type}_8_0.6.log.res"
        if not os.path.exists(log_path):
            print(f"Error: evaluation of exec_type = {exec_type}, cv = 8, req_rate = 0.6 not completed.")
            exit(1)
        handler = open(log_path, 'r')
        lines = handler.readlines()
        mapping = {}
        for line in lines:
            match = re.match(pattern, line)
            if match:
                task_type = match.group(1)
                attainment = float(match.group(4))
                mapping[task_type] = attainment
        for app_name in app_real:
            result_app.append(mapping[app_name])
        all.append(result_app)

    # colors = ['tomato', 'green', 'violet', 'dimgray', 'blue', 'orange']
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    colors = ['#A40545', '#F46F44', '#FDD985', '#E9F5A1', '#7FCBA4', '#4B65AF']
    labels = ["Serverless vLLM", "ServerlessLLM", sysname, sysname + " w/ Cache"]
    lw = 2
    ms = 8
    fs = 16
    label_fs = 20

    pos = range(len(app))

    w = 0.2
    xpos = [ i*(len(labels)+1)*w for i in range(len(all[0]))]

    for i in range(0, len(all)):
        pos = [t+i*w for t in xpos]
        rect = plt.bar(pos, all[i], width=w,color=colors[i], edgecolor='black',label = labels[i])

    ax.set_xlabel('Applications', fontsize=label_fs)
    # latency
    # ax.set_ylabel('Latency per Token (ms)', fontsize=label_fs)
    # memory
    ax.set_ylabel('SLO Attainment (%)', fontsize=label_fs)
    # ax.set_ylim(ymin=40, ymax=100)
    # ax.set_title("(a) TTFT SLO attainment", y=-0.3, fontsize=fs)

    plt.yticks(fontsize=fs)
    plt.xticks([t+1.5*w for t in xpos], app, fontsize=fs)

    bwith = 1
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)

    # fig.legend(labels, loc='upper center', ncol=2, fontsize=fs, frameon=False)
    fig.legend(labels, bbox_to_anchor=(0.93,1.1), ncol=2, fontsize=fs, frameon=False)

    plt.savefig('figs/figure11.pdf', bbox_inches='tight', pad_inches=0.2)

fig_app()