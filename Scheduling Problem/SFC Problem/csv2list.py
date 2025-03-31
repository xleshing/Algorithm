import csv



class csv2list:
    def nodes(self, file: str) -> list:
        network_nodes = []
        with open(file, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                node_id = row['節點名稱'].replace("EN", "")
                vnf_types_raw = row['VNF類型'].replace("VNF", "").split(", ")
                neighbors = row['鄰居'].replace("EN", "").split(", ")
                loads = list(map(float, row['每種VNF負載'].split(", ")))
                delays = list(map(float, row['每種VNF處理延遲'].split(", ")))

                load_per_vnf = {vnf: load for vnf, load in zip(vnf_types_raw, loads)}
                delay_per_vnf = {vnf: delay for vnf, delay in zip(vnf_types_raw, delays)}

                network_nodes.append({
                    'id': node_id,
                    'vnf_types': vnf_types_raw,
                    'neighbors': neighbors,
                    'load_per_vnf': load_per_vnf,
                    'processing_delay': delay_per_vnf
                })

        return network_nodes

    def edges(self, file: str) -> dict:
        edges = {}

        with open(file, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                start = row['邊的起點'].replace("EN", "")
                end = row['邊的終點'].replace("EN", "")
                capacity = float(row['邊的容量'])  # 根據需要可以轉成 int

                edges[(start, end)] = capacity

        return edges

    def vnfs(self, file: str) -> dict:
        vnf_traffic = {}

        with open(file, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                vnf_id = row['VNF名稱'].replace("VNF", "")
                demand = float(row['VNF流量需求'])  # 也可以用 int()
                vnf_traffic[vnf_id] = demand

        return vnf_traffic

    def demands(self, file: str) -> list:
        sfc_requests = []

        with open(file, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)

            for i, row in enumerate(reader):
                vnf_id = row['需求名稱'].replace("D", "")
                chain = row['需求鏈路'].split(',')
                chain = [vnf.replace("VNF", "").strip() for vnf in chain]  # 去除空白
                sfc_requests.append({
                    'id': vnf_id,
                    'chain': chain
                })

        return sfc_requests


if __name__ == "__main__":
    csv2list = csv2list()
    print(csv2list.nodes("problem/nodes/nodes_15.csv"))
    print(csv2list.edges("problem/edges/edges_15.csv"))
    print(csv2list.vnfs("problem/vnfs/vnfs_15.csv"))
    print(csv2list.demands("problem/demands/demands.csv"))
