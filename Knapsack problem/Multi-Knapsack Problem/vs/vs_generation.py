from NSCO_LoadBalancing import NSCO_LoadBalancing
from NSGA4_LoadBalancing import NSGA4_LoadBalancing
import relative_igd_for_algorithms

import pandas as pd

import numpy as np
# 與 NSGA4 共用的三個目標函式 (簽名與 NSGA4 一致)
def objective_latency(solution, usage, item_values, server_capacities):
    load_ratios = usage.flatten()/server_capacities.flatten()
    return np.max(load_ratios)

def objective_cost(solution, usage, item_values, server_capacities):
    load_ratios = usage.flatten()/server_capacities.flatten()
    return np.sum(load_ratios**2)

def objective_resource_utilization(solution, usage, item_values, server_capacities):
    load_ratios = usage.flatten()/server_capacities.flatten()
    return np.std(load_ratios)

def save_generation_results(algo_name, generation_pareto_fronts, ns_algo, filename=None):
    """
    把每代 Pareto 前沿的目標值存成 CSV
    algo_name: "NSGA4" or "NSCO"
    generation_pareto_fronts: evolve() 第二個回傳值
    ns_algo: 已初始化好的演算法物件 (有 fitness 方法)
    filename: 指定輸出檔名，不給的話自動生成
    """
    generation_objectives_data = []
    for gen_idx, front in enumerate(generation_pareto_fronts):
        obj_data = []
        for sol in front:
            obj_vals = ns_algo.fitness(sol)
            obj_data.append({
                'LoadBalance': float(obj_vals[0]),
                'Average Delay': float(obj_vals[1]),
                'Cost': float(obj_vals[2]),
            })
        generation_objectives_data.append({
            "Generation": gen_idx,
            "sol": obj_data
        })

    df = pd.DataFrame(generation_objectives_data)
    if filename is None:
        filename = f"{algo_name}_generation_solutions.csv"
    df.to_csv(filename, index=False)
    print(f"[{algo_name}] 已輸出到 {filename}")
    return df

objectives = [objective_resource_utilization, objective_latency, objective_cost]

# 共同的資料
num_servers   = 15
num_requests  = 10
population    = 20
generations   = 100
item_values   = np.random.randint(1, 5, (num_requests, 1))
server_caps   = np.random.randint(50, 61, (num_servers, 1))

# NSGA-4
nsga4 = NSGA4_LoadBalancing(num_servers, num_requests, population, generations,
                            objectives, item_values, server_caps)
pf_nsga4, hist_nsga4 = nsga4.evolve()

# NSCO（新版、介面對齊）
nsco = NSCO_LoadBalancing(num_servers, num_requests, population, generations,
                          objectives, item_values, server_caps,
                          coyotes_per_group=5, n_groups=4, p_leave=0.1)
pf_nsco, hist_nsco = nsco.evolve()

df_nsga4 = save_generation_results("NSGA4", hist_nsga4, nsga4)
df_nsco  = save_generation_results("NSCO", hist_nsco, nsco)
