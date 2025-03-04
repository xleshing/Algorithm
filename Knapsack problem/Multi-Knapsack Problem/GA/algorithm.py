import numpy as np
import random
import matplotlib.pyplot as plt
import os
from init_papulation import Init_population


class GeneticAlgorithm:
    def __init__(self, values: np.array, max_weight: np.array, old_value: np.array, dim, particle=100, Elite_num=40,
                 CrossoverRate=0.9, MutationRate=0.1,
                 MaxIteration=100):
        self.dim = dim
        self.values = values
        self.max_weight = max_weight
        self.knapsack_num = max_weight.shape[1]
        self.particle_num = particle
        self.item_num = self.values.shape[1]
        self.elite_num = int((Elite_num / 100) * self.particle_num)
        self.cr = CrossoverRate
        self.mr = MutationRate
        self.max_iter = MaxIteration
        self.best_solution = None
        self.best_fitness = float("Inf")
        self.fitness_values = np.array([])
        self.best_fitness_list = []
        self.old_value = old_value
        self.init_population = Init_population(self.dim, self.item_num, self.values, self.max_weight, self.knapsack_num,
                                               self.old_value, self.particle_num)

    def fitness_value(self, solution, knapsack, dim):
        total_value = np.sum(np.array(self.values[dim]) * np.array(solution)) + self.old_value[dim][knapsack]
        return total_value

    def selection(self, population, fitness_values) -> list:
        elite_index = np.argsort(fitness_values)[:self.elite_num]
        elite_parent = [population[idx] for idx in elite_index]
        return elite_parent

    def crossover(self, parent1, parent2):
        # 1. 隨機選擇一個交配點（範圍從1到物品數減一）
        crossover_point = random.randint(1, self.item_num - 1)

        # 2. 初始化兩個子代的染色體，與父代的形狀相同，但值設為零
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)

        # 3. 交換基因的區間
        # 子代1繼承父代1的基因到交配點，之後繼承父代2的基因
        child1[:, :crossover_point] = parent1[:, :crossover_point]
        child1[:, crossover_point:] = parent2[:, crossover_point:]

        # 子代2繼承父代2的基因到交配點，之後繼承父代1的基因
        child2[:, :crossover_point] = parent2[:, :crossover_point]
        child2[:, crossover_point:] = parent1[:, crossover_point:]

        # 4. 返回兩個子代
        return child1, child2

    def mutate(self, child):
        # 遍歷每個物品（行）
        for item in range(child.shape[1]):
            # 檢查該物品（行）是否有變異發生
            if random.random() < self.mr:
                # 確保在該物品行中只有一個位置可以為 1，其餘為 0
                # 隨機選擇一個新的背包索引
                new_knapsack = random.randint(0, child.shape[0] - 1)

                # 將該物品行的所有值設為 0
                child[:, item] = 0

                # 在隨機選擇的背包位置將值設為 1
                child[new_knapsack, item] = 1
        return child

    def check_fitness(self, fitness):
        for dim in range(self.dim):
            for each_knapsack in range(self.knapsack_num):
                if fitness[dim][each_knapsack] > self.max_weight[dim][each_knapsack]:
                    fitness[dim][0] = np.sum(self.max_weight[dim])
                    fitness[dim][-1] = -np.sum(self.max_weight[dim])
                    break

        return fitness

    def genetic_algorithm(self):
        population = self.init_population.init_population(self.particle_num)
        population = self.init_population.init_population_dim(population, self.particle_num)
        population = self.init_population.fix_population(population)

        for iter in range(self.max_iter):
            knapsack_fitness_values = np.zeros(shape=[self.particle_num, self.dim, self.knapsack_num])
            for p_index in range(self.particle_num):
                for dim in range(self.dim):
                    for knapsack in range(self.knapsack_num):
                        knapsack_fitness_values[p_index][dim][knapsack] = self.fitness_value(
                            population[p_index][knapsack], knapsack, dim)

            self.fitness_values = np.array([np.sum(
                [np.std((np.array(fitness[dim]) / np.array(self.max_weight[dim])).tolist(), ddof=0) for dim in
                 range(self.dim)]) for fitness in knapsack_fitness_values])
            elite_parents = self.selection(population, self.fitness_values)
            new_population = []
            for i in range(self.elite_num):
                for j in range(i, self.elite_num):
                    parent1, parent2 = elite_parents[i], elite_parents[j]
                    child1, child2 = self.crossover(parent1, parent2)
                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)
                    new_population.extend([child1, child2])
            population = population + new_population

            new_knapsack_fitness_values = np.zeros(shape=[len(population), self.dim, self.knapsack_num])
            for new_p_index in range(len(population)):
                for dim in range(self.dim):
                    for new_knapsack in range(self.knapsack_num):
                        new_knapsack_fitness_values[new_p_index][dim][new_knapsack] = self.fitness_value(
                            population[new_p_index][new_knapsack], new_knapsack, dim)

            for fitness in range(len(new_knapsack_fitness_values)):
                new_knapsack_fitness_values[fitness] = self.check_fitness(new_knapsack_fitness_values[fitness])

            fitness_values = np.array([np.sum(
                [np.std((np.array(fitness[dim]) / np.array(self.max_weight[dim])).tolist(), ddof=0) for dim in
                 range(self.dim)]) for fitness in new_knapsack_fitness_values])
            idx_to_keep = np.argsort(fitness_values)[:self.particle_num]
            population = [population[idx] for idx in idx_to_keep]
            self.best_fitness_list.append(np.min(self.fitness_values))
            if np.min(self.fitness_values) < self.best_fitness:
                self.best_solution = population[np.argmin(self.fitness_values)]
                self.best_fitness = np.min(self.fitness_values)

            os.system('cls' if os.name == 'nt' else 'clear')
            print(self.best_fitness, iter)

        return self.best_solution, self.best_fitness

    def show(self):
        plt.plot(self.best_fitness_list)
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.title('GA Best Fitness: [{:.5f}]'.format(self.best_fitness))
        plt.show()
