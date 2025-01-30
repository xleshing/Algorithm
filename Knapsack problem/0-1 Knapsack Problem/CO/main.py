import subprocess
import json
import time
from MMCOA import Algorithm
import numpy as np


### Step 1: 取得目前 Pod 使用率 ###
def get_pod_usage():
    """ 取得所有 Pod 使用的 CPU 和記憶體總和 """
    cmd = "kubectl top pod --all-namespaces --no-headers | awk '{print $2, $3}'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    total_cpu = 0
    total_mem = 0
    for line in result.stdout.strip().split("\n"):
        if line:
            cpu, mem = line.split()
            total_cpu += int(cpu.replace("m", "")) / 1000  # 轉換為核心數
            total_mem += int(mem.replace("Mi", ""))  # 轉換為 MB
    return total_cpu, total_mem


### Step 2: 取得目前節點資源上限 ###
def get_node_capacity():
    """ 取得所有節點的 CPU/記憶體上限 """
    cmd = "kubectl get nodes -o json"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    nodes = {}
    data = json.loads(result.stdout)
    # for item in data["items"]:
    #     name = item["metadata"]["name"]
    #     capacity = item["status"]["capacity"]
    #     cpu_limit = int(capacity["cpu"])  # 核心數
    #     mem_limit = int(capacity["memory"].replace("Ki", "")) // 1024  # 轉換為 MB
    #     nodes[name] = (cpu_limit, mem_limit)
    for item in data["items"]:
        name = item["metadata"]["name"]
        capacity = item["status"]["capacity"]
        cpu_limit = int(capacity["cpu"])  # 核心數
        nodes[name] = cpu_limit
    return nodes


### Step 4: 根據演算法輸出調整節點狀態 ###
def adjust_nodes(capacity, max_delay):
    pod_cpu, pod_mem = get_pod_usage()
    nodes = get_node_capacity()

    weight = [pod_cpu, pod_mem]  # 目前總負載
    values = [cpu for cpu in nodes.values()]  # 各 Node 限制

    node_list = list(nodes.keys())  # 節點名稱列表
    turn_node_on = 0

    if weight[0] / np.sum(values) * 100 > capacity:
        turn_node_on = 1

    algorithm = Algorithm(
        turn_node_on,
        d=len(values),
        value=values,
        weight=weight,
        capacity=capacity,
        coyotes_per_group=5,
        n_groups=5,
        p_leave=0.001,
        max_iter=100,
        max_delay=max_delay
    )
    best_sol, best_fit, curve = algorithm.MMCO_main()

    if best_fit > capacity:
        print("無法以目標值調整集群，請修改目標值或增加Delay")
    else:
        decision = best_sol
        print(decision)
        for i, node in enumerate(node_list):
            if decision[i] == 0:  # 如果該節點應該關閉
                print(f"讓節點 {node} 進入睡眠模式")
                # subprocess.run(f"kubectl drain {node} --ignore-daemonsets --delete-emptydir-data", shell=True)
                # subprocess.run(f"kubectl cordon {node}", shell=True)
            elif decision[i] == 1:  # 如果該節點應該啟用
                print(f"喚醒節點 {node}")
                # subprocess.run(f"kubectl uncordon {node}", shell=True)


if __name__ == "__main__":
    adjust_nodes(80, 100)
