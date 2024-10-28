import numpy as np


class Answer:
    def __init__(self, c, p, o):
        with open(c, "r") as file:
            self.lines_c = file.readlines()
        with open(p, "r") as file:
            self.lines_p = file.readlines()
        with open(o, "r") as file:
            self.lines_o = file.readlines()

    def answer(self):
        max_weight = [[float(each_line.strip()) for each_line in line.split()] for line in self.lines_c]
        values = [[float(each_line.strip()) for each_line in line.split()] for line in self.lines_p]
        old_values = [[float(each_line.strip()) for each_line in line.split()] for line in self.lines_o]
        return [np.array(max_weight).T, np.array(values).T, np.array(old_values).T]
