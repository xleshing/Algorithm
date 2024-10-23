from algorithm import PSO
import numpy as np
import matplotlib.pyplot as plt
from func import func

knapsack = [700, 500, 600, 800]
item = [52, 70, 73, 77, 78, 80, 80, 82, 88, 90, 94, 98, 106, 111, 121]
# item = [52, 70, 73, 77]

func = func(knapsack, item)

def check(sol):
    return func.fitness_value(sol)

p = 1000

iters = 100

x_max = len(knapsack) ** len(item) - 1

x_min = 0
optimizer = PSO(funct=check, num_particle=p, max_iter=iters, x_max=x_max, x_min=x_min, knapsack=knapsack)
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
