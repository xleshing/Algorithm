import numpy as np

class Answer:
    def __init__(self, c, p, w, s):
        with open(c, "r") as file:
            self.lines_c = file.readlines()
        with open(p, "r") as file:
            self.lines_p = file.readlines()
        with open(w, "r") as file:
            self.lines_w = file.readlines()
        with open(s, "r") as file:
            self.lines_s = file.readlines()

    def answer(self):
        max_weight = [int(line.strip()) for line in self.lines_c]
        values = [int(line.strip()) for line in self.lines_p]
        weights = [int(line.strip()) for line in self.lines_w]
        ans = np.array([int(line.strip()) for line in self.lines_s])
        ans_total_value = np.sum(np.array(values) * np.array(ans))
        return [values, weights, max_weight, ans, ans_total_value]
