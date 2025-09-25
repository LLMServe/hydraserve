import os
import math
import random
from typing import List, Dict, Tuple

from resource_manager import AllocatedResource
from scheduler import ModelStats
from scheduler import KVCacheRequiredSize

'''
Algorithms Return: Tuple[chosen_node_ids, gpu_usage_of_each_node, net_usage_of_each_node]
'''

max_pp_size = int(os.getenv("MAX_PP_SIZE", "4"))

'''
naive: use pp_size=1 and place model to a random node that has enough resource.
'''
def naive(rgpus: List[int], rnets: List[float], stats: ModelStats, required_pp_size: int, max_waiting_time: int = 0) -> AllocatedResource:
    model_size = stats.model_size
    required_size = math.ceil(model_size + KVCacheRequiredSize)
    num_nodes = len(rgpus)
    # avail_nodes = []
    for node_id in range(num_nodes):
        if rgpus[node_id] >= required_size:
            return AllocatedResource(1, [node_id], [required_size], [0])
    #         avail_nodes.append(node_id)
    # if len(avail_nodes) > 0:
    #     rand_id = random.randint(0, len(avail_nodes)-1)
    #     return AllocatedResource(1, [avail_nodes[rand_id]], [required_size], [0])
    return AllocatedResource(0, [], [], [])

'''
pipeline_parallel_static: use a fixed pp_size and perform load balance (try to distribute cold start pods).
'''
def pipeline_parallel_static(rgpus: List[int], rnets: List[float], stats: ModelStats, required_pp_size: int, max_waiting_time: int = 0, pp_size: int = 2) -> AllocatedResource:
    model_size = stats.model_size

    if pp_size >= 2 and model_size <= 10:
        # For small models, use pp_size=1
        pp_size = 1

    required_size = math.ceil((model_size + KVCacheRequiredSize) / pp_size)
    num_nodes = len(rgpus)

    if num_nodes < pp_size:
        return AllocatedResource(0, [], [], [])
    
    mx_net_remained = [0] * pp_size
    mx_net_ids = [-1] * pp_size

    for node_id in range(num_nodes):
        if rgpus[node_id] >= required_size:
            for index in range(pp_size):
                if rnets[node_id] > mx_net_remained[index]:
                    for i in range(pp_size - 1, index, -1):
                        mx_net_remained[i] = mx_net_remained[i-1]
                        mx_net_ids[i] = mx_net_ids[i-1]
                    mx_net_remained[index] = rnets[node_id]
                    mx_net_ids[index] = node_id
                    break
    
    if mx_net_ids[pp_size-1] != -1:
        return AllocatedResource(pp_size, mx_net_ids, [required_size] * pp_size, [min(mx_net_remained)] * pp_size)
    return AllocatedResource(0, [], [], [])

'''
pipeline_parallel_static: use a fixed pp_size and perform load balance (try to distribute cold start pods).
Reserve memory for the first node to allow scaling down
'''
def pipeline_parallel_static_scale_down(rgpus: List[int], rnets: List[float], stats: ModelStats, required_pp_size: int, max_waiting_time: int = 0, pp_size: int = 2) -> AllocatedResource:
    if pp_size == 1:
        return pipeline_parallel_static(rgpus, rnets, stats, required_pp_size, max_waiting_time, pp_size)
    
    model_size = stats.model_size

    required_size = math.ceil((model_size + KVCacheRequiredSize) / pp_size)
    master_required_size = math.ceil(model_size + KVCacheRequiredSize)
    num_nodes = len(rgpus)

    if num_nodes < pp_size:
        return AllocatedResource(0, [], [], [])
    
    # First, find a node for master
    mx_net_remained = 0
    mx_net_id = -1
    for node_id in range(num_nodes):
        if rgpus[node_id] >= master_required_size:
            if rnets[node_id] > mx_net_remained:
                mx_net_remained = rnets[node_id]
                mx_net_id = node_id

    if mx_net_id == -1:
        return AllocatedResource(0, [], [], [])

    # Second, find nodes for remained pipeline parallel stages
    mx_net_remained = [mx_net_remained] + [0] * (pp_size - 1)
    mx_net_ids = [mx_net_id] + [-1] * (pp_size - 1)

    for node_id in range(num_nodes):
        if node_id != mx_net_ids[0]:
            if rgpus[node_id] >= required_size:
                for index in range(1, pp_size):
                    if rnets[node_id] > mx_net_remained[index]:
                        for i in range(pp_size - 1, index, -1):
                            mx_net_remained[i] = mx_net_remained[i-1]
                            mx_net_ids[i] = mx_net_ids[i-1]
                        mx_net_remained[index] = rnets[node_id]
                        mx_net_ids[index] = node_id
                        break
    
    if mx_net_ids[pp_size-1] != -1:
        return AllocatedResource(pp_size, mx_net_ids,
                                 [master_required_size] + [required_size] * (pp_size - 1),
                                 [mx_net_remained[0]] + [min(mx_net_remained[1:])] * (pp_size - 1),
                                 True)
    return AllocatedResource(0, [], [], [])

def ours(rgpus: List[int], rnets: List[float], stats: ModelStats, required_pp_size: int, max_waiting_time: int) -> AllocatedResource:
    model_size = stats.model_size

    # if stats.num_profile_points > 0 and stats.tpot >= stats.tpot_slo:
    #     print(f"Ours failed: record tpot = {stats.tpot}, slo = {stats.tpot_slo}")

    check_tpot = True
    if stats.num_profile_points == 0:
        check_tpot = False
    elif stats.tpot >= stats.tpot_slo:
        return pipeline_parallel_static(rgpus, rnets, stats, required_pp_size, 0, 1)
    
    required_size = [math.ceil((model_size + KVCacheRequiredSize) / pp_size) for pp_size in range(1, max_pp_size + 1)]
    master_required_size = required_size[0]
    num_nodes = len(rgpus)

    # Calculate maximum bandwidth for master node
    num_avail_nodes = [0] * max_pp_size     # how many nodes have available gpu for each pp_size
    net_list = []
    for pp_size in range(max_pp_size):
        net_list.append([])
    for node_id in range(num_nodes):
        if rnets[node_id] > 0 and rgpus[node_id] > 0:
            mn_num_worker = max_pp_size
            if rgpus[node_id] >= master_required_size:
                mn_num_worker = 0
            else:
                for num_worker in range(1, mn_num_worker):
                    if rgpus[node_id] >= required_size[num_worker]:
                        mn_num_worker = num_worker
                        break
            for num_worker in range(mn_num_worker, max_pp_size):
                num_avail_nodes[num_worker] += 1
                net_list[num_worker].append(rnets[node_id])
    
    for pp_size in range(max_pp_size):
        net_list[pp_size].sort(reverse=True)
    
    if num_avail_nodes[0] == 0:
        return AllocatedResource(0, [], [], [])

    # Find the lowest ttft
    cur_cost = stats.get_cur_cost()
    mn_ttft = 0
    mn_ttft_pp_size = -1
    mn_ttft_num_all_card = -1
    for pp_size in range(1, max_pp_size + 1):
        if mn_ttft_pp_size >= required_pp_size:
            # There has an optimal solution
            break
        if num_avail_nodes[pp_size-1] < pp_size:
            # There is no enough nodes even if all instances only use minimum required size (actually the master node will use all card).
            continue
        if pp_size <= required_pp_size:
            # let all workers use all-card
            mn_num_all_card = pp_size
        else:
            mn_num_all_card = required_pp_size

        for num_all_card in range(mn_num_all_card, pp_size + 1):
            if num_all_card > num_avail_nodes[0]:
                break
            tpot_inc_rate = pp_size - num_all_card + num_all_card / pp_size
            if check_tpot:
                predicted_tpot = stats.tpot * tpot_inc_rate + 1 * (pp_size - 1)
                if predicted_tpot > stats.tpot_slo:
                    continue
            
            # Get maxmimum bandiwidth
            if pp_size == 1:
                band = net_list[0][0]
            else:
                band = min(net_list[0][num_all_card-1], net_list[pp_size-1][pp_size-1])

            # predict how many resource we will consume before scale-down
            # scale_down_time = model_size * (pp_size - 1) / pp_size / band * 8
            # scale_down_cost = scale_down_time * (num_all_card * master_required_size + (pp_size - num_all_card) * required_size[pp_size-1])

            # check whether the cost satisfies cost_slo (for faster auto-scaling, we ignore cost limit)
            # if (scale_down_cost + cur_cost) / stats.num_output_tokens <= stats.cost_slo or required_pp_size >= 2 and num_all_card >= required_pp_size:

            # Predict cold start time (note that for prefill time, we ignore the network transfer time cost)
            # NOTE: Approximation. The time cost after model pulling (model loading, initialization, etc.) is 2 second
            prefill_time = 2 if check_tpot else stats.prefill_time
            ttft = model_size / pp_size / band * 8 * 1000.0 + prefill_time * tpot_inc_rate + max_waiting_time + 2000
            
            # check whether the ttft satisfies ttft_slo
            if ttft < stats.ttft_slo:
                mn_ttft = ttft
                mn_ttft_pp_size = pp_size
                mn_ttft_num_all_card = num_all_card
                print(f"ALG: predicted ttft = {ttft}, pp_size = {pp_size}, all_card = {num_all_card}, required = {required_pp_size}")
                break

    if mn_ttft_pp_size == -1:
        # fall back to pp_size=1
        # print('Ours failed: fall back to pp_size=1')
        return pipeline_parallel_static(rgpus, rnets, stats, required_pp_size, 0, 1)
    
    # Allocation
    # Now we have (mn_ttft_num_all_card - 1) all-card and (mn_ttft_pp_size - mn_ttft_num_all_card) part-card to allocate
    pp_size = mn_ttft_pp_size
    num_all_card = mn_ttft_num_all_card
    required_size = required_size[pp_size-1]
    mx_net_remained = [0] * pp_size
    mx_net_ids = [-1] * pp_size
    for node_id in range(num_nodes):
        if rnets[node_id] > 0 and rgpus[node_id] >= required_size:
            if rgpus[node_id] >= master_required_size:
                start_pos = 0
            else:
                start_pos = num_all_card
            for index in range(start_pos, pp_size):
                if rnets[node_id] > mx_net_remained[index]:
                    # Do list shift
                    # TODO: use two list. use list insert function. make it simple.
                    if index < num_all_card:
                        if num_all_card < pp_size and mx_net_remained[num_all_card-1] > mx_net_remained[pp_size-1]:
                            # Insert the last one of master list into worker list
                            for i in range(pp_size - 2, num_all_card - 1, -1):
                                if mx_net_remained[num_all_card-1] > mx_net_remained[i]:
                                    mx_net_remained[i+1] = mx_net_remained[i]
                                    mx_net_ids[i+1] = mx_net_ids[i]
                                else:
                                    mx_net_remained[i+1] = mx_net_remained[num_all_card-1]
                                    mx_net_ids[i+1] = mx_net_ids[num_all_card-1]
                                    break
                            else:
                                # No break happens, meaning that the new one is the largest
                                mx_net_remained[num_all_card] = mx_net_remained[num_all_card-1]
                                mx_net_ids[num_all_card] = mx_net_ids[num_all_card-1]
                        # Move master list
                        for i in range(num_all_card - 1, index, -1):
                            mx_net_remained[i] = mx_net_remained[i-1]
                            mx_net_ids[i] = mx_net_ids[i-1]
                    else:
                        # Move worker list
                        for i in range(pp_size - 1, index, -1):
                            mx_net_remained[i] = mx_net_remained[i-1]
                            mx_net_ids[i] = mx_net_ids[i-1]
                    mx_net_remained[index] = rnets[node_id]
                    mx_net_ids[index] = node_id
                    break
    
    if mx_net_ids[num_all_card-1] != -1 and mx_net_ids[pp_size-1] != -1:
        return AllocatedResource(pp_size, mx_net_ids,
                                 [master_required_size] * num_all_card + [required_size] * (pp_size - num_all_card),
                                 mx_net_remained,
                                 True if pp_size > 1 and num_all_card < pp_size else False)
    
    print(f"Error: alg: no resource available for {pp_size} and num_all_card = {num_all_card}, rgpus = {rgpus}, rnets = {rnets}, mx_net_ids = {mx_net_ids}")

    return AllocatedResource(0, [], [], [])
