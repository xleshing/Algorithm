import numpy as np

class Answer:
    def __init__(self, c, p):
        with open(c, "r") as file:
            self.lines_c = file.readlines()
        with open(p, "r") as file:
            self.lines_p = file.readlines()

    def answer(self):
        max_weight = [int(line.strip()) for line in self.lines_c]
        values = [int(line.strip()) for line in self.lines_p]
        return [values, max_weight]
