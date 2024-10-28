import numpy as np

class Test:
    def __init__(self, sol, func, dim):
        self.answer = func
        self.sol = sol
        self.dim = dim

    def print_answer(self):
        c = np.array(self.answer.answer()[0])
        p = np.array(self.answer.answer()[1])
        o = np.array(self.answer.answer()[2])

        sol = np.array(self.sol)
        ans = []
        index = 0
        for each_sol in sol:
            ans.append([["{:.5%}".format((np.sum(each_sol * p[dim]) + o[dim][index]) / c[dim][index]), np.sum(each_sol * p[dim]) + o[dim][index], c[dim][index]] for dim in range(self.dim)])
            index += 1
        print(np.array(ans))
