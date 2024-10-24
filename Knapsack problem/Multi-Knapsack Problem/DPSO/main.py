from algorithm import PSO
import numpy as np
import matplotlib.pyplot as plt
from func import func
from data import Answer

a = Answer("p07_c.txt", "p07_p.txt", "p07_o.txt")
knapsack = a.answer()[1]
item = a.answer()[0]
old_value = a.answer()[2]

func = func(knapsack, item, old_value)


def check(sol):
    return func.fitness_value(sol)


x_max = len(knapsack) ** len(item) - 1

x_min = 0
optimizer = PSO(funct=check, num_particle=1000, max_iter=100, x_max=x_max, x_min=x_min, knapsack=knapsack)
sol = optimizer.update()
print(func.change_sol(sol).tolist())
optimizer.plot_curve()

X_list = optimizer.X_history
V_list = optimizer.V_history
fig, ax = plt.subplots(1, 1)
ax.set_title('title', loc='center')
line = ax.plot([], [], 'b.')

X_grid, Y_grid = np.meshgrid(np.linspace(-1.0, 1.0, 40), np.linspace(-1.0, 1.0, 40))
Z_grid = np.zeros((40, 40))
for i in range(Z_grid.shape[1]):
    Z_grid[:, i] = check(np.vstack([X_grid[:, i], Y_grid[:, i]]).T)
ax.contour(X_grid, Y_grid, Z_grid, 20)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
plt.ion()
plt.show()
