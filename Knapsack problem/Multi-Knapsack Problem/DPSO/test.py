from data import Answer
import numpy as np

class Test:
    def __init__(self, a, func):
        self.answer = func
        self.a = a

    def print_answer(self):
        c = np.array(self.answer.get_data()[0])
        w = np.array(self.answer.get_data()[1])
        o = np.array(self.answer.get_data()[2])

        a = np.array(self.a)
        ans = []
        index = 0
        for b in a:
            ans.append(["{:.5%}".format((np.sum(b * w) + o[index]) / c[index]), np.sum(b * w) + o[index], c[index]])
            index += 1
        print(np.array(ans))
