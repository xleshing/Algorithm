from kubernetes import client, config
import json
import numpy as np
import argparse
import logging
from logging.handlers import RotatingFileHandler
from MMCOA import Algorithm


# 初始化 Kubernetes API
# 在 Kubernetes 內部執行時改用 `config.load_incluster_config()`
try:
    config.load_incluster_config()  # 讓 Pod 內部可以存取 API Server
except:
    config.load_kube_config()  # 如果在本機測試，則載入 kubeconfig

v1 = client.CoreV1Api()


def evict_pod(pod_name, namespace):
    """ 優雅地驅逐 Pod，實現 `kubectl drain` 效果 """
    eviction_body = client.V1Eviction(
        metadata=client.V1ObjectMeta(name=pod_name, namespace=namespace),
        delete_options=client.V1DeleteOptions(grace_period_seconds=30)  # 給 30 秒優雅關閉
    )
    try:
        v1.create_namespaced_pod_eviction(name=pod_name, namespace=namespace, body=eviction_body)
        logger.info(f"{pod_name}（Evict）")
    except Exception as e:
        logger.warning(f"{pod_name}（Evict Failed）：{e}")

def drain_node(node):
    """ 執行 `kubectl drain` 效果 """
    # 1️⃣ `cordon` 節點（標記為不可調度）
    v1.patch_node(node, {"spec": {"unschedulable": True}})
    logger.info(f"{node}（Cordon）")

    # 2️⃣ 找到該節點上的 Pod
    pods = v1.list_pod_for_all_namespaces(field_selector=f"spec.nodeName={node}").items

    # 3️⃣ 驅逐所有非 DaemonSet 的 Pod
    for pod in pods:
        is_evict = True
        if pod.metadata.owner_references:
            for owner in pod.metadata.owner_references:
                if owner.kind == "DaemonSet":
                    logger.info(f"Skip DaemonSet Pod：{pod.metadata.name}")
                    is_evict = False  # DaemonSet 不應該被刪除

        # 使用 Eviction API 來優雅遷移 Pod
        if is_evict:
            evict_pod(pod.metadata.name, pod.metadata.namespace)

def uncordon_node(node):
    v1.patch_node(node, {"spec": {"unschedulable": False}})
    logger.info(f"{node}（Uncordon）")

# 取得目前節點資源上限
def get_node_capacity():
    """ 取得所有 **非 Master** 節點的 CPU/記憶體上限，並判斷是否在休眠狀態（cordon 狀態）"""
    nodes = []
    values = []
    status = []

    # 取得所有節點
    all_nodes = v1.list_node()

    for node in all_nodes.items:
        name = node.metadata.name

        # 檢查是否為 Master 節點
        labels = node.metadata.labels
        is_master = "node-role.kubernetes.io/control-plane" in labels

        if is_master:
            logger.debug(f"⚠️ 跳過 Master 節點: {name}")
            continue

            # 取得 CPU/Memory 上限
        capacity = node.status.capacity
        cpu_limit = int(capacity["cpu"])
        mem_limit = int(capacity["memory"].replace("Ki", "")) // 1024

        # 檢查是否是 "休眠" 狀態（cordon 狀態）
        unschedulable = node.spec.unschedulable if node.spec.unschedulable else False
        node_status = 0 if unschedulable else 1

        nodes.append(name)
        values.append(cpu_limit)
        status.append(node_status)

    return nodes, values, status

# 根據演算法輸出調整節點狀態
def adjust_nodes(pre_pod_cpu, pre_pod_mem, capacity, active_range, max_delay):
    turn_node_on = 0
    pod_cpu, pod_mem = pre_pod_cpu, pre_pod_mem
    node_list, values, node_status = get_node_capacity()

    weight = [pod_cpu, pod_mem]

    # logger.debug("所有 pod 總消耗（核）：", weight[0], "所有可用 node：", node_list, "各 node CPU 上限（核）：", values, "目前總負載（％）：", weight[0] / np.dot(node_status, values) * 100)
    logger.debug(f"所有 pod 總消耗（核）：{weight[0]}，所有可用 node：{node_list}，各 node CPU 上限（核）：{values}，目前總負載（％）：{weight[0] / np.dot(node_status, values) * 100:.2f}")
    if weight[0] / np.dot(node_status, values) * 100 + active_range < capacity or weight[0] / np.dot(node_status, values) * 100 > capacity:
        if weight[0] / np.dot(np.ones_like(node_status), values) * 100 > capacity:
            logger.warning(f"Not enough resources, will activate all node（Target Value：{capacity} %，The Cluster Load after activate all node：{weight[0] / np.dot(np.ones_like(node_status), values) * 100} %）")
            for node in node_list:
                uncordon_node(node)  # 使用 `uncordon` API
            logger.info(f"Nodes Status：{np.zeros_like(node_list, dtype=int).tolist()}, Cluster Load：{weight[0] / np.dot(np.zeros_like(node_list, dtype=int).tolist(), values) * 100}%")
            logger.info("----")
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
                logger.debug("目前已是最佳或找不到最佳，保持原狀態")
            else:
                decision = best_sol
                # 先開再關
                for i, node in enumerate(node_list):
                    if decision[i] == 1 and node_status[i] == 0:  # 需要開啟且目前是休眠狀態
                        uncordon_node(node)  # 使用 `uncordon` API

                for i, node in enumerate(node_list):
                    if decision[i] == 0 and node_status[i] == 1:  # 需要關閉且目前是可用狀態
                        drain_node(node)  # 使用 `drain` API

                logger.info(f"Nodes Status：{decision}, Cluster Load：{weight[0] / np.dot(decision, values) * 100}%")
                logger.info("----")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kubernetes Node Scaler")
    parser.add_argument("--pre_pod_cpu", type=float, help="Predict Pod CPU Usage（％％）")
    parser.add_argument("--pre_pod_mem", type=float, help="Predict Pod Memery Usage（％％）")
    parser.add_argument("--capacity", type=float, default=80.0, help="Target Resource Usage（％％）")
    parser.add_argument("--activate_range", type=float, default=10.0, help="Upper bound - Lower bound（％％）")
    parser.add_argument("--max_calculate_times", type=int, default=100, help="Max Calculate Times")
    parser.add_argument("--log_level", type=str, default="INFO", help="Log Level（DEBUG, INFO, WARNING, ERROR, CRITICAL）")

    args = parser.parse_args()
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)

    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    log_handler = RotatingFileHandler("Node_Scaler.log", maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
    log_handler.setFormatter(log_formatter)

    # ✅ 同時輸出到終端機（讓 `kubectl logs` 可見）
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)

    logger = logging.getLogger("Node_Scaler_logger")
    logger.setLevel(log_level)
    logger.addHandler(log_handler)
    logger.addHandler(console_handler)  # ✅ 記錄到命令行（stdout）

    logger.info(f"Target Resource Usage：{args.capacity}%, Upper bound - Lower bound：{args.activate_range}%, Max Calculate Times：{args.max_calculate_times}, Log Level：{args.log_level}")

    adjust_nodes(args.pre_pod_cpu, args.pre_pod_mem, args.capacity, args.activate_range, args.max_calculate_times)
