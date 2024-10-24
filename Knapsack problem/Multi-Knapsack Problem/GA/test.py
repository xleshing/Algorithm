from data import Answer
import numpy as np

answer = Answer("p07_c.txt", "p07_p.txt", "p07_o.txt")

c = np.array(answer.answer()[1])
w = np.array(answer.answer()[0])
o = np.array(answer.answer()[2])

a =  [[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
a = np.array(a)
ans = []
index = 0
for b in a:
    ans.append([(np.sum(b * w) + o[index]) / c[index], np.sum(b * w) + o[index], c[index]])
    index += 1
print(np.array(ans))

