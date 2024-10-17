from algorithm import PSO

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from func import func

processing_times = np.array([
    [54, 9, 38, 95],
    [34, 15, 19, 34],
    [61, 89, 28, 7],
    [2, 70, 87, 29]])

machines = np.array([
    [2, 0, 3, 1],
    [3, 0, 1, 2],
    [0, 1, 2, 3],
    [0, 2, 1, 3]])

func = func(machines.shape[0], machines.shape[1], processing_times=processing_times)

def check(n=machines):
    return func.check(n)

p = 1000

d = machines.shape[0]
iters = 100

x_max = 23 * np.ones(d)

x_min = 0 * np.ones(d)

optimizer = PSO(funct=check, num_dim=d, num_particle=p, max_iter=iters, x_max=x_max, x_min=x_min, c1=10, c2=10, k=2)
optimizer.update()
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
p = plt.show()

def update_scatter(frame):
    i, j = frame // 10, frame % 10
    ax.set_title('iter = ' + str(i))
    X_tmp = X_list[i] + V_list[i] * j / 10.0
    plt.setp(line, 'xdata', X_tmp[:, 0], 'ydata', X_tmp[:, 1])
    return line

ani = FuncAnimation(fig, update_scatter, blit=False, interval=25, frames=500)
ani.save('pso.gif', writer='pillow', fps=20)