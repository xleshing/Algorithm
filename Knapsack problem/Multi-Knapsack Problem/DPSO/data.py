
class Answer:
    def __init__(self, c, p, o):
        with open(c, "r") as file:
            self.lines_c = file.readlines()
        with open(p, "r") as file:
            self.lines_p = file.readlines()
        with open(o, "r") as file:
            self.lines_o = file.readlines()

    def answer(self):
        max_weight = [float(line.strip()) for line in self.lines_c]
        values = [float(line.strip()) for line in self.lines_p]
        old_values = [float(line.strip()) for line in self.lines_o]
        return [max_weight, values, old_values]
