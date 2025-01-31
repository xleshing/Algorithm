import subprocess
import json
import time
from MMCOA import Algorithm
import numpy as np
import argparse
import logging
from logging.handlers import RotatingFileHandler



def convert_memory_to_mib(mem_str):
    """將記憶體字串 (如 '256Mi', '1Gi', '512Ki', '100M') 轉換為 MiB"""
    mem_str = mem_str.strip()  # 移除空格
    if mem_str.endswith("Mi"):
        return int(mem_str[:-2])  # 直接轉換為 MiB
    elif mem_str.endswith("Gi"):
        return int(mem_str[:-2]) * 1024  # Gi 轉換成 MiB
    elif mem_str.endswith("Ki"):
        return int(mem_str[:-2]) // 1024  # Ki 轉換成 MiB
    elif mem_str.endswith("M"):  # 兼容 '100M' 這種寫法
        return int(mem_str[:-1])  # 假設 100M = 100Mi
    elif mem_str.endswith("G"):  # 兼容 '1G' 這種寫法
        return int(mem_str[:-1]) * 1024  # 1G = 1024Mi
    else:
        try:
            return int(mem_str)  # 嘗試直接轉換
        except ValueError:
            logger.debug(f"⚠️ 無法解析記憶體數值: {mem_str}")
            return 0  # 如果格式錯誤，返回 0


# 取得目前 Pod 使用率
def get_pod_usage():
    """ 取得所有 **非 Master 節點** 上的 Pod 的 CPU 和記憶體使用率 """
    cmd = "kubectl get pods --all-namespaces -o json"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    total_cpu = 0  # 單位：mCPU (1 CPU = 1000m)
    total_mem = 0  # 單位：MiB

    data = json.loads(result.stdout)

    # 取得 Master 節點名稱（避免計入）
    master_nodes = set(get_node_capacity()[0])  # 取得非 Master 節點
    all_nodes = set([item["metadata"]["name"] for item in data["items"]])
    master_nodes = all_nodes - master_nodes  # 取得 Master 節點

    for pod in data["items"]:
        try:
            node_name = pod["spec"].get("nodeName", "")
            if node_name in master_nodes:
                continue  # 跳過 Master 節點上的 Pod

            containers = pod["spec"]["containers"]
            for container in containers:
                resources = container.get("resources", {})
                requests = resources.get("requests", {})

                cpu = requests.get("cpu", "0m")
                mem = requests.get("memory", "0Mi")

                # 轉換 CPU（mCPU）
                if cpu.endswith("m"):
                    cpu = int(cpu[:-1]) / 1000
                else:
                    cpu = int(cpu)  # 無 "m" 代表完整核心數

                # 轉換記憶體（MiB）
                mem = convert_memory_to_mib(mem)

                total_cpu += cpu
                total_mem += mem
        except KeyError:
            continue  # 如果 Pod 沒有 CPU/Memory 指標，則跳過

    return total_cpu, total_mem  # CPU 轉換成核心數


# 取得目前節點資源上限
def get_node_capacity():
    """ 取得所有 **非 Master** 節點的 CPU/記憶體上限，並判斷是否在休眠狀態（cordon 狀態）"""
    cmd = "kubectl get nodes -o json"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    nodes = []       # 節點名稱
    values = []      # 節點資源 (CPU, Memory)
    status = []      # 0 = 休眠, 1 = 可用 (cordon 狀態)

    data = json.loads(result.stdout)
    for item in data["items"]:
        name = item["metadata"]["name"]

        # 檢查是否為 Master 節點
        labels = item["metadata"].get("labels", {})
        taints = item["spec"].get("taints", [])

        is_master = "node-role.kubernetes.io/control-plane" in labels
        # has_no_schedule = any(t["effect"] == "NoSchedule" for t in taints)

        if is_master:
            logger.debug(f"⚠️ 跳過 Master 節點: {name}")
            continue  # 跳過 Master 節點

        # 取得 CPU/Memory 上限
        capacity = item["status"]["capacity"]
        cpu_limit = int(capacity["cpu"])  # 核心數
        mem_limit = int(capacity["memory"].replace("Ki", "")) // 1024  # 轉換為 MiB

        # 檢查是否是 "休眠" 狀態（是否被 cordon）
        unschedulable = item["spec"].get("unschedulable", False)
        node_status = 0 if unschedulable else 1  # 0 = 休眠, 1 = 可用

        nodes.append(name)
        # values.append((cpu_limit, mem_limit))
        values.append(cpu_limit)
        status.append(node_status)

    return nodes, values, status  # 回傳 (節點名稱, 資源上限, 休眠狀態)


# 根據演算法輸出調整節點狀態
def adjust_nodes(capacity, active_range, max_delay):
    turn_node_on = 0
    pod_cpu, pod_mem = get_pod_usage()
    node_list, values, node_status = get_node_capacity()

    weight = [pod_cpu, pod_mem]  # 目前總負載

    logger.debug("所有pod總消耗（核）：", weight[0], "所有可找到 node：", node_list, "各 node CPU 上限（核）：", values, "目前總負載（％）：", weight[0] / np.dot(node_status, values) * 100)

    if weight[0] / np.dot(node_status, values) * 100 + active_range < capacity or weight[0] / np.dot(node_status, values) * 100 > capacity:

        if weight[0] / np.dot(np.ones_like(node_status), values) * 100 > capacity:
            logger.warning(f"資源過低，請調整目標值（目標值：{capacity} %，如果node全開之集群負載：{weight[0] / np.dot(np.ones_like(node_status), values) * 100} %）")
        else:
            algorithm = Algorithm(
                turn_node_on,
                d=len(values),
                value=values,
                weight=weight[0],
                capacity=capacity,
                coyotes_per_group=5,
                n_groups=5,
                p_leave=0.001,
                max_iter=100,
                max_delay=max_delay,
                original_status=node_status
            )
            best_sol, best_fit, curve = algorithm.MMCO_main()

            if best_sol.tolist() == node_status:
                logger.warning("目前已是最佳或找不到最佳，保持原狀態，或修改目標值與Max Calculate times")
            else:
                decision = best_sol
                for i, node in enumerate(node_list):
                    if decision[i] == 0 and node_status[i] == 1:  # 需要關閉且目前是可用狀態
                        logger.info(f"讓節點 {node} 進入睡眠模式")
                        subprocess.run(f"kubectl drain {node} --ignore-daemonsets --delete-emptydir-data", shell=True)
                        subprocess.run(f"kubectl cordon {node}", shell=True)

                    elif decision[i] == 1 and node_status[i] == 0:  # 需要開啟且目前是休眠狀態
                        logger.info(f"喚醒節點 {node}")
                        subprocess.run(f"kubectl uncordon {node}", shell=True)

                logger.debug("最佳解：", decision)
                logger.debug("目前總負載（％）：", weight[0] / np.dot(decision, values) * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kubernetes 節點擴充器")
    parser.add_argument("--capacity", type=float, default=80.0, help="目標資源使用率 預設80(%%)")
    parser.add_argument("--active_range", type=float, default=10.0, help="上下限空間 預設10(%%)")
    parser.add_argument("--max_calculate_times", type=int, default=100, help="最大計算次數 預設100")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="設定日誌級別（DEBUG, INFO, WARNING, ERROR, CRITICAL）")
    args = parser.parse_args()

    # 將日誌等級轉換為 logging 模組可用的值
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)

    # 設定日誌格式
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # 設定日誌輪替（最大 5MB，保留 3 份）
    log_handler = RotatingFileHandler("Node_Scaler.log", maxBytes=5*1024*1024, backupCount=3)
    log_handler.setFormatter(log_formatter)

    # 設定 logger
    logger = logging.getLogger("Node_Scaler_logger")
    logger.setLevel(log_level)  # 由外部參數決定日誌等級
    logger.addHandler(log_handler)

    adjust_nodes(args.capacity, args.active_range, args.max_calculate_times)
