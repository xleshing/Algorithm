
class Answer:
    def __init__(self, c, p, o):
        with open(c, "r") as file:
            self.lines_c = file.readlines()
        with open(p, "r") as file:
            self.lines_p = file.readlines()
        with open(o, "r") as file:
            self.lines_o = file.readlines()

    def answer(self):
        max_weight = [int(line.strip()) for line in self.lines_c]
        values = [int(line.strip()) for line in self.lines_p]
        old_values = [int(line.strip()) for line in self.lines_o]
        return [values, max_weight, old_values]
