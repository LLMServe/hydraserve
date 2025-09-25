import os
import sys
import pickle
import re
import random
from random import sample
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
import ast

from ModelInfo import ModelList

def get_requests(workload_path, req_per_second):
    with open(workload_path, 'rb') as f:
        reqs = pickle.load(f)
    
    print(f"Load {len(reqs)} requests in total.")

    mx_time_stamp = 60 * 60         # convert the one-day trace to a one-hour trace
    time_scale = (24 * 60 * 60) / mx_time_stamp

    for index, req in enumerate(reqs):
        arrival_time_in_second = req[0] / time_scale
        if arrival_time_in_second >= mx_time_stamp:
            break
    
    reqs = reqs[:index]
    cur_req_per_second = len(reqs) / mx_time_stamp

    if cur_req_per_second <= req_per_second:
        print(f"maximum req_per_second = {cur_req_per_second}, which is smaller than required. Do not change.")
    else:
        num_sample = int(req_per_second * mx_time_stamp)
        reqs = sample(reqs, num_sample)

        def takeFirst(elem):
            return elem[0]
        reqs.sort(key=takeFirst)
    
    print(f"Get {len(reqs)} requests in {mx_time_stamp} seconds")
    return reqs, time_scale, mx_time_stamp

@dataclass
class Result:
    ttft: float
    tpot: float
    tokens: int
    model: str
    model_id: int

def process_results(lines: List[str]) -> Dict[int, Result]:
    pattern = r"\[(\d+)\]: TTFT = (-?[\d\.]+)s, TPOT = (-?[\d\.]+) ms, #Tokens = (\d+) \((.+)/(\d+)\)"
    gen_pattern = r"Generate request \[(\d+)\] for model (.+)/(\d+)"
    res_dict = {}
    gen_dict = {}

    num = 0
    num_tokens = 0
    sum_ttft = 0
    sum_tpot = 0
    mx_id = 0

    for line in lines:
        match = re.match(pattern, line)
        if match:
            id = int(match.group(1))
            ttft = float(match.group(2))
            tpot = float(match.group(3))
            tokens = int(match.group(4))
            model = match.group(5)
            model_id = int(match.group(6))

            num += 1
            num_tokens += tokens 
            sum_ttft += ttft
            sum_tpot += tpot * tokens

            res_dict[id] = Result(ttft, tpot, tokens, model, model_id)
        else:
            match = re.match(gen_pattern, line)
            if match:
                id = int(match.group(1))
                model = match.group(2)
                model_id = int(match.group(3))
                gen_dict[id] = (model, model_id)
                mx_id = max(mx_id, id)
    
    print(f"AVG TTFT = {sum_ttft / num}")
    print(f"AVG TPOT = {sum_tpot / num_tokens}")

    if num < mx_id:
        print(f"There are {mx_id-num} requests do not have results. Assume very large TTFT.")
        for id in range(1, mx_id):
            if id not in res_dict:
                if id not in gen_dict:
                    print(f"Error: id {id} not in generation dict and result dict.")
                res_dict[id] = Result(999, 0, 0, gen_dict[id][0], gen_dict[id][1])

    return res_dict

@dataclass
class InstanceInfo:
    model: str = ""
    requests: List[Tuple[int, float]] = field(default_factory=list)
    gpu_usage: List[int] = field(default_factory=list)
    init_time: float = 0
    scale_down_time: float = 0

# Return instance -> [request id] mapping.
def get_instance_info(lines: List[str]) -> Dict[int, int]:
    alloc_pattern = r"request \[(\d+)\] allocated to instance \[(\d+)\], waiting time = ([\d\.]+) seconds"
    sched_pattern = r"scheduler: create instance \[(\d+)\] for (.+)/(\d+), allocate_time = (.+) s, create_pod_time = (.+) s, init_pod_time = (.+) s"
    usage_pattern = r"GPU Usage: (.+)"
    load_dest_pattern = r"scheduler: instance \[(\d+)\] starts to load dest at ([\d\.]+) s"
    scale_down_compl_pattern = r"scheduler: scale down instance \[(\d+)\], time cost = ([\d\.]+) seconds"
    scale_up_pattern = r"scheduler: create new instance \[(\d+)\] from \[(\d+)\], time cost = ([\d\.]+) seconds"

    instance_info = {}

    for line in lines:
        match = re.match(sched_pattern, line)
        if match:
            ins_id = int(match.group(1))
            model = match.group(2)
            model_id = int(match.group(3))
            create_pod_time = float(match.group(5))
            init_pod_time = float(match.group(6))
            model_ = model + "/" + str(model_id)
            instance_info[ins_id] = InstanceInfo(model=model_, init_time=create_pod_time+init_pod_time)
            continue
        
        match = re.match(usage_pattern, line)
        if match:
            # variable $ins_id stores the last instance id
            gpu_usage = ast.literal_eval(match.group(1))
            instance_info[ins_id].gpu_usage = gpu_usage
            continue
        
        match = re.match(load_dest_pattern, line)
        if match:
            ins_id = int(match.group(1))
            wait_time = float(match.group(2))
            if ins_id in instance_info:
                instance_info[ins_id].scale_down_time = wait_time
            continue
        
        match = re.match(scale_down_compl_pattern, line)
        if match:
            ins_id = int(match.group(1))
            compl_time = float(match.group(2))
            if ins_id in instance_info:
                instance_info[ins_id].scale_down_time += compl_time
            continue
        
        match = re.match(scale_up_pattern, line)
        if match:
            new_ins_id = int(match.group(1))
            old_ins_id = int(match.group(2))
            create_time = float(match.group(3))
            info = instance_info[old_ins_id]
            instance_info[new_ins_id] = InstanceInfo(model=info.model, gpu_usage=[info.gpu_usage[0]])
            continue

        match = re.match(alloc_pattern, line)
        if match:
            req_id = int(match.group(1))
            ins_id = int(match.group(2))
            wait_time = float(match.group(3))
            instance_info[ins_id].requests.append((req_id, wait_time))
            continue
        
    return instance_info

if __name__ == '__main__':
    random.seed(51)

    result_file = sys.argv[1]
    if len(sys.argv) >= 3:
        if len(sys.argv) != 5:
            print("To analyze cost, provide main_file, workload_path, and req_per_second.")
            exit(1)
        main_file = sys.argv[2]
        workload_path = sys.argv[3]
        req_per_second = float(sys.argv[4])
    else:
        main_file = None

    result_handler = open(result_file, 'r')
    lines = result_handler.readlines()
    res_dict = process_results(lines)
    num_request = len(res_dict.keys())
    result_handler.close()

    model_task_type = {}      # model -> task type
    model_slo = {}            # model -> (ttft_slo, tpot_slo)
    num_reqs = {}             # task type -> num_reqs
    sum_tokens = {}           # task type -> sum_tokens
    model_sum_tokens = {}     # model -> sum_tokens (expect for fisrt token)
    model_sum_decoding = {}   # model -> sum_decoding_time
    model_sum_tpot = {}       # model -> sum_tpot
    cur_model_index = {}
    for task_type, model_dict in ModelList.items():
        sum_tokens[task_type] = 0
        num_reqs[task_type] = 0
        for model_id, model_info in model_dict.items():
            ttft_slo = model_info[1]
            tpot_slo = model_info[2]
            model_num = model_info[4]
            if model_id not in cur_model_index:
                cur_model_index[model_id] = 0
                cur_index = 0
            else:
                cur_index = cur_model_index[model_id]
            for index in range(model_num):
                model_id_ = model_id + "/" + str(index + cur_index)
                model_task_type[model_id_] = task_type
                model_slo[model_id_] = (ttft_slo, tpot_slo)
            cur_model_index[model_id] += model_num

    ttft_violation = {}
    tpot_violation = {}
    for task_type in ModelList.keys():
        ttft_violation[task_type] = 0
        tpot_violation[task_type] = 0

    for index in range(num_request):
        # Obtain TTFT and TPOT
        id = index + 1
        if id not in res_dict:
            print(f"Error: cannot find result for index {id}")
        else:
            res = res_dict[id]
            model_id = res.model + "/" + str(res.model_id)
            if model_id not in model_task_type:
                print(f"Error: index {id} has model id {model_id} which is not in model_task_type")
            else:
                # Obtain SLO
                task_type = model_task_type[model_id]
                ttft_slo, tpot_slo = model_slo[model_id]
                ttft_slo /= 1000
                num_reqs[task_type] += 1
                sum_tokens[task_type] += res.tokens
                if model_id not in model_sum_tokens:
                    model_sum_tokens[model_id] = 0
                    model_sum_decoding[model_id] = 0
                    model_sum_tpot[model_id] = 0
                model_sum_tokens[model_id] += res.tokens - 1
                model_sum_decoding[model_id] += res.tpot * (res.tokens - 1)
                model_sum_tpot[model_id] += res.tpot

                if res.ttft > ttft_slo:
                    # print(f"req_id: {id}, ttft = {res.ttft}, model_id: {model_id}")
                    ttft_violation[task_type] += 1
                if res.tpot > tpot_slo:
                    tpot_violation[task_type] += 1
    
    print(f"Num Request = {num_request}")
    violation = 0
    violation_tpot = 0
    num_req = 0
    for task_type in ModelList.keys():
        violation += ttft_violation[task_type]
        violation_tpot += tpot_violation[task_type]
        num_req += num_reqs[task_type]
        print(f"Task {task_type} ttft vio = {ttft_violation[task_type]}, tpot vio = {tpot_violation[task_type]}, ttft attain = { '%.1f' % ((1-ttft_violation[task_type]/num_reqs[task_type])*100)}, tpot attain = { '%.1f' % ((1-tpot_violation[task_type]/num_reqs[task_type])*100)}")#, num reqest = {num_reqs[task_type]}, num tokens = {sum_tokens[task_type] / num_reqs[task_type]}")
    print(f"total ttft violation = {violation}, attainment = {'%.1f' % ((1-violation/num_req) * 100)}")
    print(f"total tpot violation = {violation_tpot}, attainment = {'%.1f' % ((1-violation_tpot/num_req) * 100)}")

    if main_file:
        # Analyse cost
        # Note that only check resource usage in the cluster is not accurate, because the resource usage during cold start is not charged
        print("Start analyzing cost...")

        # 1. Get the arrival time of requests
        random.seed(51)
        reqs, time_scale, mx_time_stamp = get_requests(workload_path, req_per_second)

        if len(reqs) != len(res_dict.keys()):
            print(f"Generator has {len(reqs)} requests to generate, but there are only {len(res_dict.keys())} requests in the result log.")
            exit(1)

        # 2. Obtain instance information
        main_handler = open(main_file, 'r')
        lines = main_handler.readlines()
        instance_info = get_instance_info(lines)
        main_handler.close()

        # 3. Get the resource cost of each instance
        grace_period = 30
        model_cost = {}
        model_num_reqs = {}
        for ins_id, info in instance_info.items():
            if info.requests:
                mn_creation_time = 99999999
                mx_end_time = 0
                for req_id, wait_time in info.requests:
                    arrival_time = reqs[req_id-1][0] / time_scale
                    instance_creation_time = arrival_time + wait_time
                    mn_creation_time = min(mn_creation_time, instance_creation_time)

                    inference_end_time = arrival_time + res_dict[req_id].ttft + res_dict[req_id].tpot * res_dict[req_id].tokens / 1000.0
                    mx_end_time = max(mx_end_time, inference_end_time)
                instance_start_create_time = mn_creation_time - info.init_time
                instance_end_time = mx_end_time + grace_period
                instance_alive_interval = instance_end_time - instance_start_create_time
            else:
                # This instance do not meet any request, so it exists for $grace_period
                instance_alive_interval = grace_period
            total_gpu_usage = 0
            for usage in info.gpu_usage:
                total_gpu_usage += usage
            if info.scale_down_time > 1 and info.scale_down_time < instance_alive_interval:
                resource_usage = info.scale_down_time * total_gpu_usage + (instance_alive_interval - info.scale_down_time) * info.gpu_usage[0]
            else:
                resource_usage = instance_alive_interval * total_gpu_usage
            if info.model not in model_cost:
                model_cost[info.model] = 0
                model_num_reqs[info.model] = 0
            model_cost[info.model] += resource_usage
            model_num_reqs[info.model] += len(info.requests)

        # 4. Get average cost for each task type
        sum_cost = {}           # task type -> sum_cost
        num_model = {}
        for task_type, model_dict in ModelList.items():
            sum_cost[task_type] = 0
            num_model[task_type] = 0
        for model, cost in model_cost.items():
            task_type = model_task_type[model]
            sum_cost[task_type] += cost
            # sum_cost[task_type] += cost / model_num_reqs[model]
            # num_model[task_type] += 1

        # 5. Outputs
        for task_type in ModelList.keys():
            avg_cost = sum_cost[task_type] / num_reqs[task_type]
            # avg_cost = sum_cost[task_type] / num_model[task_type]
            print(f"Task {task_type} avg cost = {avg_cost}")
        
        avg_cost = 0
        for model, cost in model_cost.items():
            avg_cost += cost
        print(f"Average cost = {avg_cost / num_request}")

        def getKey(model_name: str):
            pos = model_name.rfind("/")
            model_str = model_name[:pos]
            model_id = int(model_name[pos+1:])
            return (model_task_type[model_name], model_str, model_id)
        cost_file = os.getenv("COST_LOG", "")
        if cost_file:
            with open(cost_file, "w") as f:
                for model, cost in sorted(model_cost.items(), key=lambda item: getKey(item[0])):
                    print(f"{model_task_type[model]},{model},{cost/model_num_reqs[model]}", file=f)

        tot_sum_tokens = 0
        tot_sum_decoding = 0
        for task_type in ModelList.keys():
            sum_tokens = 0
            sum_decoding = 0
            for model, task_type_ in model_task_type.items():
                if model in model_sum_tokens and task_type_ == task_type:
                    sum_tokens += model_sum_tokens[model]
                    sum_decoding += model_sum_decoding[model]
            avg_tpot = sum_decoding / sum_tokens
            print(f"Task {task_type} avg tpot = {avg_tpot}")
            tot_sum_tokens += sum_tokens
            tot_sum_decoding += sum_decoding
        print(f"Average tpot = {tot_sum_decoding / tot_sum_tokens}")

        tpot_file = os.getenv("TPOT_LOG", "")
        if tpot_file:
            with open(tpot_file, "w") as f:
                for model, cost in sorted(model_cost.items(), key=lambda item: getKey(item[0])):
                    print(f"{model_task_type[model]},{model},{model_sum_decoding[model]/model_sum_tokens[model]}", file=f)
