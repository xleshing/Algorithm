from kubernetes import client, config
from kubernetes.client import CustomObjectsApi
import json
import numpy as np
import argparse
import logging
from logging.handlers import RotatingFileHandler
from nsco import NSCO_Algorithm
import time

# 初始化 Kubernetes API
# 在 Kubernetes 內部執行時改用 `config.load_incluster_config()`
try:
    config.load_incluster_config()  # 讓 Pod 內部可以存取 API Server
except:
    config.load_kube_config()  # 如果在本機測試，則載入 kubeconfig

v1 = client.CoreV1Api()


def load(x, w, v):
    ratio = w / np.dot(x, v) * 100
    return 1 / (ratio + 1e-6)


def change(x, original_status):
    return np.sum(np.abs(x - original_status))


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


def drain_node(node, namespaces):
    """ 執行 `kubectl drain` 效果 """
    # 1️⃣ `cordon` 節點（標記為不可調度）
    v1.patch_node(node, {"spec": {"unschedulable": True}})
    logger.info(f"{node}（Cordon）")

    # 2️⃣ 找到該節點上的 Pod
    all_pods = []
    for ns in namespaces:
        pods = v1.list_namespaced_pod(namespace=ns, field_selector=f"spec.nodeName={node}").items
        all_pods.extend(pods)  # 將結果累加

    # 3️⃣ 驅逐所有非 DaemonSet 的 Pod
    for pod in all_pods:
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


def convert_memory_to_mib(mem_str):
    """將記憶體字串 (如 '256Mi', '1Gi', '512Ki', '100M') 轉換為 MiB"""
    mem_str = mem_str.strip()
    if mem_str.endswith("Mi"):
        return int(mem_str[:-2])
    elif mem_str.endswith("Gi"):
        return int(mem_str[:-2]) * 1024
    elif mem_str.endswith("Ki"):
        return int(mem_str[:-2]) // 1024
    elif mem_str.endswith("M"):
        return int(mem_str[:-1])
    elif mem_str.endswith("G"):
        return int(mem_str[:-1]) * 1024
    else:
        try:
            return int(mem_str)
        except ValueError:
            logger.error(f"Unable to parse memory value: {mem_str}")
            return 0

        # 取得目前 Pod 使用率


def get_pod_usage(namespaces, mode="requests"):
    """根據 mode 決定如何抓取 pod 資源使用狀態：
       - 'requests'：用 requests 來估算
       - 'metrics'：用 metrics.k8s.io API
       - 'hybrid'：requests 為 0 時 fallback 到 metrics"""

    total_cpu = 0
    total_mem = 0

    master_nodes = set(get_node_capacity()[0])
    all_pods = []
    for ns in namespaces:
        pods = v1.list_namespaced_pod(namespace=ns).items
        all_pods.extend(pods)

    # 如果是 hybrid 或 metrics 模式，先拉 Metrics 資料
    metrics_data = {}
    if mode in ["metrics", "hybrid"]:
        try:
            metrics_api = CustomObjectsApi()
            for ns in namespaces:
                metrics = metrics_api.list_namespaced_custom_object(
                    group="metrics.k8s.io",
                    version="v1beta1",
                    namespace=ns,
                    plural="pods"
                )
                for item in metrics["items"]:
                    pod_name = item["metadata"]["name"]
                    metrics_data[(ns, pod_name)] = item
        except Exception as e:
            logger.warning(f"無法取得 Metrics 資料：{e}")
            if mode == "metrics":
                return 0, 0  # Metrics 模式下沒資料就直接返回 0

    for pod in all_pods:
        try:
            node_name = pod.spec.node_name
            if node_name in master_nodes:
                continue

            cpu_sum = 0
            mem_sum = 0

            for container in pod.spec.containers:
                requests = container.resources.requests or {}
                cpu = requests.get("cpu", "0m")
                mem = requests.get("memory", "0Mi")

                if cpu.endswith("m"):
                    cpu_val = int(cpu[:-1]) / 1000
                else:
                    try:
                        cpu_val = int(cpu)
                    except:
                        cpu_val = 0

                mem_val = convert_memory_to_mib(mem)

                cpu_sum += cpu_val
                mem_sum += mem_val

            if cpu_sum == 0 and mode == "hybrid":
                # fallback to metrics data
                metrics_item = metrics_data.get((pod.metadata.namespace, pod.metadata.name))
                if metrics_item:
                    for container in metrics_item["containers"]:
                        usage_cpu = container["usage"]["cpu"]
                        usage_mem = container["usage"]["memory"]

                        if usage_cpu.endswith("n"):
                            cpu_val = int(usage_cpu[:-1]) / 1e9
                        elif usage_cpu.endswith("u"):
                            cpu_val = int(usage_cpu[:-1]) / 1e6
                        elif usage_cpu.endswith("m"):
                            cpu_val = int(usage_cpu[:-1]) / 1000
                        else:
                            cpu_val = int(usage_cpu)

                        mem_val = convert_memory_to_mib(usage_mem)

                        cpu_sum += cpu_val
                        mem_sum += mem_val

            elif mode == "metrics":
                metrics_item = metrics_data.get((pod.metadata.namespace, pod.metadata.name))
                if metrics_item:
                    cpu_sum = 0
                    mem_sum = 0
                    for container in metrics_item["containers"]:
                        usage_cpu = container["usage"]["cpu"]
                        usage_mem = container["usage"]["memory"]

                        if usage_cpu.endswith("n"):
                            cpu_val = int(usage_cpu[:-1]) / 1e9
                        elif usage_cpu.endswith("u"):
                            cpu_val = int(usage_cpu[:-1]) / 1e6
                        elif usage_cpu.endswith("m"):
                            cpu_val = int(usage_cpu[:-1]) / 1000
                        else:
                            cpu_val = int(usage_cpu)

                        mem_val = convert_memory_to_mib(usage_mem)

                        cpu_sum += cpu_val
                        mem_sum += mem_val

            total_cpu += cpu_sum
            total_mem += mem_sum

        except KeyError:
            continue

    return total_cpu, total_mem
# def get_pod_usage(namespaces):
#     """ 取得所有 **非 Master 節點** 上的 Pod 的 CPU 和記憶體使用率 """
#     total_cpu = 0
#     total_mem = 0
#
#     # 取得 Master 節點列表
#     master_nodes = set(get_node_capacity()[0])
#
#     # 獲取所有 Pod
#     all_pods = []
#     for ns in namespaces:
#         pods = v1.list_namespaced_pod(namespace=ns).items
#         all_pods.extend(pods)  # 將所有 Pod 加入列表
#
#     for pod in all_pods:
#         try:
#             node_name = pod.spec.node_name
#             if node_name in master_nodes:
#                 continue
#
#             for container in pod.spec.containers:
#                 requests = container.resources.requests or {}
#
#                 cpu = requests.get("cpu", "0m")
#                 mem = requests.get("memory", "0Mi")
#
#                 if cpu.endswith("m"):
#                     cpu = int(cpu[:-1]) / 1000
#                 else:
#                     cpu = int(cpu)
#
#                 mem = convert_memory_to_mib(mem)
#
#                 total_cpu += cpu
#                 total_mem += mem
#         except KeyError:
#             continue
#
#     return total_cpu, total_mem


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
            logger.debug(f"跳過 Master 節點: {name}")
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
def adjust_nodes(capacity, tolerance_value, jitter_tolerance, max_delay, namespaces_str, mode):
    namespaces = namespaces_str.split(" ")
    turn_node_on = 0
    pod_cpu, pod_mem = get_pod_usage(namespaces, mode)
    node_list, values, node_status = get_node_capacity()

    weight = [pod_cpu, pod_mem]

    logger.debug(
        f"所有 pod 總消耗（核）：{weight[0]}，所有可用 node：{node_list}，各 node CPU 上限（核）：{values}，目前總負載（％）：{weight[0] / np.dot(node_status, values) * 100:.2f}")
    if weight[0] / np.dot(node_status, values) * 100 + tolerance_value < capacity or weight[0] / np.dot(node_status,
                                                                                                        values) * 100 - tolerance_value > capacity:
        if weight[0] / np.dot(np.ones_like(node_status), values) * 100 > capacity:
            logger.warning(
                f"Not enough resources, will activate all node（Target Value：{capacity} %，The Cluster Load after activate all node：{weight[0] / np.dot(np.ones_like(node_status), values) * 100} %）")
            for i, node in enumerate(node_list):
                if node_status[i] == 0:
                    uncordon_node(node)  # 使用 `uncordon` API
            logger.info(
                f"Nodes Status：{np.ones_like(node_list, dtype=int).tolist()}, Cluster Load：{weight[0] / np.dot(np.ones_like(node_list, dtype=int).tolist(), values) * 100}%")
            logger.info("----")
        else:
            algorithm = NSCO_Algorithm(
                turn_node_on=turn_node_on,
                d=len(values),
                value=values,
                weight=weight[0],
                capacity=capacity,
                coyotes_per_group=5,
                n_groups=5,
                p_leave=0.001,
                max_iter=100,
                max_delay=max_delay,
                original_status=node_status,
                objs_func=[load, change]
            )
            best_pf, _ = algorithm.NSCO_main()

            best_pf_filter = [b_pf for b_pf in best_pf if abs((weight[0] / np.dot(node_status, values) * 100) - (weight[0] / np.dot(b_pf, values) * 100)) >= jitter_tolerance]

            if best_pf_filter:
                best_ind = sorted(range(len(best_pf_filter)),
                                  key=lambda k: [change(sol, np.array(node_status)) for sol in best_pf_filter][k])
                best_sol = np.array(best_pf_filter[best_ind[0]])
            else:
                best_sol = np.array(node_status)

            if best_sol.tolist() == node_status or abs((weight[0] / np.dot(node_status, values) * 100) - (weight[0] / np.dot(best_sol, values) * 100)) <= jitter_tolerance:
                logger.debug(f"目前已是最佳或找不到最佳，保持原狀態{best_sol.tolist() == node_status, abs((weight[0] / np.dot(node_status, values) * 100) - (weight[0] / np.dot(best_sol, values) * 100)), best_sol.tolist(), node_status}")
            else:
                decision = best_sol
                # 先開再關
                for i, node in enumerate(node_list):
                    if decision[i] == 1 and node_status[i] == 0:  # 需要開啟且目前是休眠狀態
                        uncordon_node(node)  # 使用 `uncordon` API

                for i, node in enumerate(node_list):
                    if decision[i] == 0 and node_status[i] == 1:  # 需要關閉且目前是可用狀態
                        drain_node(node, namespaces)  # 使用 `drain` API

                logger.info(f"Nodes Status：{decision}, Cluster Load：{weight[0] / np.dot(decision, values) * 100}%")
                logger.info("----")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kubernetes Node Scaler")
    parser.add_argument("--namespaces", type=str, default="default",
                        help="Target Pod Namespaces (namespace_1 namespace_2 ...)")
    parser.add_argument("--capacity", type=float, default=80.0, help="Target Resource Usage（％％）")
    parser.add_argument("--tolerance_value", type=float, default=10.0, help="Tolerance Value（％％）")
    parser.add_argument("--jitter_tolerance", type=float, default=10.0, help="Jitter Tolerance Value（％％）")
    parser.add_argument("--max_calculate_times", type=int, default=100, help="Max Calculate Times")
    parser.add_argument("--sleep_time", type=int, default=5, help="Sleep Time（s）")
    parser.add_argument("--mode", type=str, choices=["requests", "metrics", "hybrid"], default="hybrid",
                        help="Resource collection mode: requests, metrics, or hybrid")
    parser.add_argument("--log_level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO",
                        help="Log Level（DEBUG, INFO, WARNING, ERROR, CRITICAL）")

    args = parser.parse_args()
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)

    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    log_handler = RotatingFileHandler("Node_Scaler.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    log_handler.setFormatter(log_formatter)

    # ✅ 同時輸出到終端機（讓 `kubectl logs` 可見）
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)

    logger = logging.getLogger("Node_Scaler_logger")
    logger.setLevel(log_level)
    logger.addHandler(log_handler)
    logger.addHandler(console_handler)  # ✅ 記錄到命令行（stdout）

    logger.info(
        f"Target Pod Namespaces：{args.namespaces}, Target Resource Usage：{args.capacity}%, Tolerance Value：{args.tolerance_value}%, Max Calculate Times：{args.max_calculate_times}, Sleep Time：{args.sleep_time}s, Log Level：{args.log_level}")
    # print(f"目標資源使用率：{args.capacity}%, 上下限空間：{args.active_range}%, 最大計算次數：{args.max_calculate_times}, 睡眠時間{args.sleep_time}s, 日誌級別：{args.log_level}")
    while True:
        adjust_nodes(capacity=args.capacity,
                     tolerance_value=args.tolerance_value,
                     jitter_tolerance=args.jitter_tolerance,
                     max_delay=args.max_calculate_times,
                     namespaces_str=args.namespaces,
                     mode=args.mode)
        time.sleep(args.sleep_time)
