import numpy as np


class print_solution:
    def __init__(self, sol, func, dim):
        self.get_data = func
        self.sol = sol
        self.dim = dim

    def print_solution(self):
        c = np.array(self.get_data.get_data()[0])
        p = np.array(self.get_data.get_data()[1])
        o = np.array(self.get_data.get_data()[2])

        sol = np.array(self.sol)
        ans = []
        index = 0
        for each_sol in sol:
            ans.append([["knapsack {}, label{}".format(index, dim), "{:.5%}".format((np.sum(each_sol * p[dim]) + o[dim][
                index]) / c[dim][index])] for dim in range(self.dim)])
            index += 1
        print(np.array(ans))
