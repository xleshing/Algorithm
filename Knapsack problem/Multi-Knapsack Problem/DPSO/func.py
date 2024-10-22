import numpy as np
import math


class func:
    def __init__(self, knapsack, iterm):
        self.max_weight = knapsack
        self.values = iterm

    def get_permutation(self, k):
        n = len(self.max_weight)
        permutation = []

        for i in range(n):
            fact = n ** (n - 1 - i)  # 計算當前位的可能組合數
            index = k // fact
            permutation.append(range(n)[index])
            k %= fact

        return permutation

    def fitness_value(self, solution, knapsack):
        sol = np.zeros(shape=[len(self.max_weight), len(self.values)])
        solution = (self.get_permutation(solution))
        for
            for each_knapsack_sol in solution:
                solution = sol[each_knapsack_sol]
        total_value = np.sum(np.array(self.values) * np.array(solution))
        if total_value > self.max_weight[knapsack]:
            return 0
        else:
            return total_value

