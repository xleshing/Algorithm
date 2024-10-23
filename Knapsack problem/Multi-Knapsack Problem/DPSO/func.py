import numpy as np


class func:
    def __init__(self, knapsack, iterm):
        self.max_weight = knapsack
        self.values = iterm

    def get_permutation(self, index):
        knapsack_num = len(self.max_weight)
        item_num = len(self.values)
        permutation = []
        k = index  # 使用輸入的序號

        for i in range(item_num):
            fact = knapsack_num ** (item_num - 1 - i)  # 計算當前位的可能組合數
            idx = k // fact  # 找到當前位的索引
            permutation.append(list(range(knapsack_num))[idx])
            k %= fact  # 更新剩下的k值

        return permutation

    def change_sol(self, solution):
        sol = np.zeros(shape=[len(self.max_weight), len(self.values)])
        solution = (self.get_permutation(solution))
        for each_knapsack_sol in range(len(solution)):
            sol[solution[each_knapsack_sol], each_knapsack_sol] = 1

        return sol

    def fitness_value(self, solution):
        solution = int(solution)
        sol = self.change_sol(solution)
        total_value = [np.sum(np.array(self.values) * np.array(s)) for s in sol]

        for values in range(len(total_value)):
            if total_value[values] > self.max_weight[values]:
                total_value[values] = self.max_weight[values]
        return np.array(total_value)


if __name__ == "__main__":
    f = func([100, 100, 100, 100], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(f.fitness_value(0))
