import matplotlib.pyplot as plt
import numpy as np
import os
import random

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sysname = 'HydraServe'

serverless_vllm_path = "/root/logs/expr_1_serverless_vllm_8_0.6.log.res"
hydraserve_path = "/root/logs/expr_1_hydraserve_8_0.6.log.res"

def fig_cost_detail():
    plt.clf()
    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.add_subplot(1, 1, 1)

    # Read results
    cur_num_task_type = 0
    task_types = {}
    task_dict_naive = {}
    cost_dict_naive = []
    with open(serverless_vllm_path + ".cost", "r") as f:
        lines = f.readlines()
        for line in lines:
            res = line.split(',')
            task_type = res[0]
            model = res[1]
            cost = float(res[2])
            if task_type not in task_types:
                task_types[task_type] = cur_num_task_type
                cur_num_task_type += 1
            task_dict_naive[model] = task_types[task_type]
            cost_dict_naive.append((model, cost))
    
    task_dict_nocache = {}
    cost_dict_nocache = []
    with open(hydraserve_path + ".cost", "r") as f:
        lines = f.readlines()
        for line in lines:
            res = line.split(',')
            task_type = res[0]
            model = res[1]
            cost = float(res[2])
            task_dict_nocache[model] = task_types[task_type]
            cost_dict_nocache.append((model, cost))
    
    assert len(cost_dict_naive) == len(cost_dict_nocache)
    
    all_xpos = [[], [], []]
    all_y = [[], [], []]
    all_heights = [[], [], []]
    all_bottoms = [[], [], []]

    cur_pos = 0
    group_gap = 0
    min_ratio = 3
    max_ratio = 0
    for index in range(len(cost_dict_naive)):
        model = cost_dict_naive[index][0]
        naive_cost = cost_dict_naive[index][1]
        nocache_cost = cost_dict_nocache[index][1]
        ratio = nocache_cost / naive_cost
        min_ratio = min(min_ratio, ratio)
        max_ratio = max(max_ratio, ratio)
        if index > 0 and task_dict_naive[model] != task_dict_naive[cost_dict_naive[index-1][0]]:
            cur_pos += group_gap
        all_xpos[task_dict_naive[model]].append(cur_pos)
        all_y[task_dict_naive[model]].append(ratio)
        if ratio >= 1:
            all_heights[task_dict_naive[model]].append(ratio - 1)
            all_bottoms[task_dict_naive[model]].append(1)
        else:
            all_heights[task_dict_naive[model]].append(1 - ratio)
            all_bottoms[task_dict_naive[model]].append(ratio)
        cur_pos += 1

    labels = ['Chatbot', 'Code', 'Summarization']
    
    colors = ['#A40545', '#F46F44', '#FDD985', '#E9F5A1', '#7FCBA4', '#4B65AF']
    lw = 2
    ms = 8
    fs = 16
    label_fs = 20

    w = 1

    for type in range(cur_num_task_type):
        # y=1 as axis
        # rect = plt.bar(all_xpos[type], all_heights[type], bottom=all_bottoms[type], width=w, color=colors[type])
        
        # y=0 as axis
        rect = plt.bar(all_xpos[type], all_y[type], width=w, label=labels[type], color=colors[type])

    ax.set_xlabel('Model ID', fontsize=label_fs)
    ax.set_ylabel('Cost Ratio', fontsize=label_fs)
    ax.set_ylim(ymin=0, ymax=1.7)
    ax.set_xlim(xmin=-2, xmax=cur_pos+1)
    # ax.set_title("(a) TTFT SLO attainment", y=-0.3, fontsize=fs)

    plt.yticks(fontsize=fs)

    bwith = 1
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)

    # fig.legend(labels, bbox_to_anchor=(0.90,1), ncol=3, fontsize=fs, frameon=False)

    ax.axhline(y=1, color='black', linestyle='--', alpha=0.7, linewidth=1)

    plt.savefig('figs/figure13-b.pdf', bbox_inches='tight', pad_inches=0.2)

def fig_tpot_detail():
    plt.clf()
    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.add_subplot(1, 1, 1)

    # Read results
    cur_num_task_type = 0
    task_types = {}
    task_dict_naive = {}
    cost_dict_naive = []
    with open(serverless_vllm_path + ".tpot", "r") as f:
        lines = f.readlines()
        for line in lines:
            res = line.split(',')
            task_type = res[0]
            model = res[1]
            cost = float(res[2])
            if task_type not in task_types:
                task_types[task_type] = cur_num_task_type
                cur_num_task_type += 1
            task_dict_naive[model] = task_types[task_type]
            cost_dict_naive.append((model, cost))
    
    task_dict_nocache = {}
    cost_dict_nocache = []
    with open(hydraserve_path + ".tpot", "r") as f:
        lines = f.readlines()
        for line in lines:
            res = line.split(',')
            task_type = res[0]
            model = res[1]
            cost = float(res[2])
            task_dict_nocache[model] = task_types[task_type]
            cost_dict_nocache.append((model, cost))
    
    assert len(cost_dict_naive) == len(cost_dict_nocache)
    
    all_xpos = [[], [], []]
    all_y = [[], [], []]
    all_heights = [[], [], []]
    all_bottoms = [[], [], []]

    cur_pos = 0
    group_gap = 0
    min_ratio = 3
    max_ratio = 0
    for index in range(len(cost_dict_naive)):
        model = cost_dict_naive[index][0]
        naive_cost = cost_dict_naive[index][1]
        nocache_cost = cost_dict_nocache[index][1]
        ratio = nocache_cost / naive_cost
        min_ratio = min(min_ratio, ratio)
        max_ratio = max(max_ratio, ratio)
        if index > 0 and task_dict_naive[model] != task_dict_naive[cost_dict_naive[index-1][0]]:
            cur_pos += group_gap
        all_xpos[task_dict_naive[model]].append(cur_pos)
        all_y[task_dict_naive[model]].append(ratio)
        if ratio >= 1:
            all_heights[task_dict_naive[model]].append(ratio - 1)
            all_bottoms[task_dict_naive[model]].append(1)
        else:
            all_heights[task_dict_naive[model]].append(1 - ratio)
            all_bottoms[task_dict_naive[model]].append(ratio)
        cur_pos += 1

    labels = ['Chatbot', 'Code', 'Summarization']
    
    colors = ['#A40545', '#F46F44', '#FDD985', '#E9F5A1', '#7FCBA4', '#4B65AF']
    lw = 2
    ms = 8
    fs = 16
    label_fs = 20

    w = 1

    for type in range(cur_num_task_type):
        # y=1 as axis
        # rect = plt.bar(all_xpos[type], all_heights[type], bottom=all_bottoms[type], width=w, color=colors[type])
        
        # y=0 as axis
        rect = plt.bar(all_xpos[type], all_y[type], width=w, label=labels[type], color=colors[type])

    ax.set_xlabel('Model ID', fontsize=label_fs)
    ax.set_ylabel('TPOT Ratio', fontsize=label_fs)
    ax.set_ylim(ymin=0, ymax=1.7)
    ax.set_xlim(xmin=-2, xmax=cur_pos+1)
    # ax.set_title("(a) TTFT SLO attainment", y=-0.3, fontsize=fs)

    plt.yticks(fontsize=fs)

    bwith = 1
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)

    # fig.legend(labels, loc='upper center', ncol=3, fontsize=fs, frameon=False)
    fig.legend(labels, bbox_to_anchor=(0.93,1), ncol=3, fontsize=fs, frameon=False)

    ax.axhline(y=1, color='black', linestyle='--', alpha=0.7, linewidth=1)

    plt.savefig('figs/figure13-a.pdf', bbox_inches='tight', pad_inches=0.2)

fig_tpot_detail()
fig_cost_detail()
