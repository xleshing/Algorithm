import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(42)

class PSO:
    def __init__(self, funct, x_max, x_min, knapsack, num_particle=20, max_iter=500, w_max=0.9, w_min=0.4, c1=2.0, c2=2.0, k=0.2):
        self.funct = funct
        self.num_particle = num_particle
        self.max_iter = max_iter
        self.x_max = x_max
        self.x_min = x_min
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.k = k
        self._iter = 1
        self.knapsack = knapsack
        self.global_best_curve = np.zeros(self.max_iter)
        self.X = np.random.randint(low=self.x_min, high=self.x_max+1, size=[self.num_particle])  # Initialize particles
        self.V = np.zeros(shape=[self.num_particle])
        self.v_max = np.array([(self.k * (self.x_max - self.x_min) / 2)])
        self.individual_best_solution = self.X.copy()
        self.individual_best_value = np.array([np.std([self.funct(position) / np.array(self.knapsack)], ddof=0) for position in self.X])
        self.global_best_solution = self.individual_best_solution[self.individual_best_value.argmin()].copy()

        self.global_best_value = self.individual_best_value.min()

        self.X_history = []
        self.V_history = []

    def update(self):
        while self._iter <= self.max_iter:
            self.X_history.append(self.X.copy())
            self.V_history.append(self.V.copy())
            R1 = np.random.uniform(size=self.num_particle)
            R2 = np.random.uniform(size=self.num_particle)
            w = self.w_max - self._iter * (self.w_max - self.w_min) / self.max_iter

            for i in range(self.num_particle):
                self.V[i] = (w * self.V[i] +
                                self.c1 * (self.individual_best_solution[i] - self.X[i]) * R1[i] +
                                self.c2 * (self.global_best_solution - self.X[i]) * R2[i])

                self.V[i] = np.where(self.V[i] > self.v_max, self.v_max, self.V[i])
                self.V[i] = np.where(self.V[i] < -self.v_max, -self.v_max, self.V[i])

                self.X[i] = self.X[i] + self.V[i]
                self.X[i] = np.where(self.X[i] > self.x_max, self.x_max, self.X[i])
                self.X[i] = np.where(self.X[i] < self.x_min, self.x_min, self.X[i])

                score = np.std([self.funct(self.X[i]) / np.array(self.knapsack)], ddof=0)


                if score < self.individual_best_value[i]:
                    self.individual_best_value[i] = score
                    self.individual_best_solution[i] = self.X[i].copy()
                    if score < self.global_best_value:
                        self.global_best_value = score
                        self.global_best_solution = self.X[i].copy()

            self.global_best_curve[self._iter - 1] = self.global_best_value

            os.system('cls' if os.name == 'nt' else 'clear')
            print(self.global_best_value, self._iter)

            self._iter += 1
        return self.global_best_solution

    def plot_curve(self):
        plt.figure()
        plt.title('PSO Loss Curve [{:.5f}]'.format(self.global_best_value))
        plt.ylabel('Best Fitness')
        plt.xlabel('Generation')
        plt.plot(self.global_best_curve, label='Loss')
        plt.grid()
        plt.legend()
        plt.show()
