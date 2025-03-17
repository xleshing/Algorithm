import numpy as np

from MMCOA import Algorithm


# 根據演算法輸出調整節點狀態
def adjust_nodes(capacity, active_range, max_delay, namespaces_str):
    namespaces = namespaces_str.split(" ")
    turn_node_on = 0
    pod_cpu, pod_mem = sum([1300, 2300, 1150, 400, 345, 765, 2323, 1231, 1212, 4002]), [0]
    node_list, values, node_status = ["master", "node1", "node2", "node3", "node4", "node5", "node6", "node7", "node8",
                                      "node9"], [8000, 8000, 8000, 8000, 8000, 8000, 8000, 8000, 8000, 8000], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    weight = [pod_cpu, pod_mem]

    print(f"所有 pod 總消耗（核）：{weight[0]}")
    print(f"所有可用 node：{node_list}，")
    print(f"各 node CPU 上限（核）：{values}，目前總負載（％）：{weight[0] / np.dot(node_status, values) * 100:.2f}")
    if weight[0] / np.dot(node_status, values) * 100 + active_range < capacity or weight[0] / np.dot(node_status,
                                                                                                     values) * 100 - active_range > capacity:
        if weight[0] / np.dot(np.ones_like(node_status), values) * 100 > capacity:
            print(
                f"Not enough resources, will activate all node（Target Value：{capacity} %，The Cluster Load after activate all node：{weight[0] / np.dot(np.ones_like(node_status), values) * 100} %）")
            for i, node in enumerate(node_list):
                if node_status[i] == 0:
                    print("uncordon_node:", node)  # 使用 `uncordon` API
            print(
                f"Nodes Status：{np.ones_like(node_list, dtype=int).tolist()}, Cluster Load：{weight[0] / np.dot(np.ones_like(node_list, dtype=int).tolist(), values) * 100}%")
            print("----")
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
                print("目前已是最佳或找不到最佳，保持原狀態")
            else:
                decision = best_sol
                # 先開再關
                for i, node in enumerate(node_list):
                    if decision[i] == 1 and node_status[i] == 0:  # 需要開啟且目前是休眠狀態
                        print("uncordon_node:", node)  # 使用 `uncordon` API

                for i, node in enumerate(node_list):
                    if decision[i] == 0 and node_status[i] == 1:  # 需要關閉且目前是可用狀態
                        print("drain_node:", node, namespaces)  # 使用 `drain` API

                print(f"Nodes Status：{decision}, Cluster Load：{weight[0] / np.dot(decision, values) * 100}%")
                print("----")


if __name__ == "__main__":
    adjust_nodes(80, 10, 100, "default")
