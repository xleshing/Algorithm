import numpy as np
import random
import matplotlib.pyplot as plt
import os

class GeneticAlgorithm:
    def __init__(self, values, max_weight, particle=100, Elite_num=40, CrossoverRate=0.9, MutationRate=0.1,
                 MaxIteration=100, knapsack_num=4):
        self.values = values
        self.max_weight = max_weight
        self.knapsack_num = knapsack_num
        self.particle_num = particle
        self.item_num = len(self.values)
        self.elite_num = Elite_num
        self.cr = CrossoverRate
        self.mr = MutationRate
        self.max_iter = MaxIteration
        self.best_solution = None
        self.best_fitness = float("Inf")
        self.fitness_values = np.array([])
        self.best_fitness_list = []

    def fitness_value(self, solution, knapsack):
        total_value = np.sum(np.array(self.values) * np.array(solution))
        if total_value > self.max_weight[knapsack]:
            return self.max_weight[knapsack]
        else:
            return total_value

    def init_population(self):
        return [np.random.randint(self.knapsack_num, size=[self.item_num]) for _ in range(self.particle_num)]

    def init_population_dim(self, population):
        sol = [np.zeros(shape=[self.knapsack_num, self.item_num]) for _ in range(self.particle_num)]
        for chromosome in range(len(population)):
            item = 0
            for knapsack in population[chromosome]:
                sol[chromosome][knapsack, item] = 1
                item += 1
        return sol


    def selection(self, population, fitness_values):
        elite_index = np.argsort(fitness_values)[:self.elite_num]
        elite_parent = [population[idx] for idx in elite_index]
        return elite_parent

    def crossover(self, parent1, parent2):
        # 1. 隨機選擇交配點的數量（至少一個，最多為物品數減一）
        num_crossover_points = random.randint(1, self.item_num - 1)

        # 2. 根據選擇的數量，在物品範圍內隨機選擇多個交配點，並進行排序
        crossover_points = sorted(random.sample(range(1, self.item_num), num_crossover_points))

        # 3. 初始化兩個子代的染色體，與父代的形狀相同，但值設為零
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)

        # 4. 開始交換基因的區間
        start = 0
        for i, point in enumerate(crossover_points):
            if i % 2 == 0:
                # 當 i 是偶數時，子代1繼承父代1的基因，子代2繼承父代2的基因
                child1[:, start:point] = parent1[:, start:point]
                child2[:, start:point] = parent2[:, start:point]
            else:
                # 當 i 是奇數時，子代1繼承父代2的基因，子代2繼承父代1的基因
                child1[:, start:point] = parent2[:, start:point]
                child2[:, start:point] = parent1[:, start:point]
            # 更新起始點為當前切點
            start = point

        # 5. 處理剩餘部分的基因（從最後一個交配點到染色體結尾）
        child1[:, start:] = parent1[:, start:]
        child2[:, start:] = parent2[:, start:]

        # 6. 返回兩個子代
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

    def genetic_algorithm(self):
        #print(self.max_weight)
        population = self.init_population()
        population = self.init_population_dim(population)
        for iter in range(self.max_iter):
            knapsack_fitness_values = np.zeros(shape=[self.particle_num, self.knapsack_num])
            for p_index in range(self.particle_num):
                for knapsack in range(self.knapsack_num):
                    knapsack_fitness_values[p_index][knapsack] = self.fitness_value(population[p_index][knapsack], knapsack)
            #print(knapsack_fitness_values)
                #print(np.std(knapsack_fitness_values[p_index], ddof=0))

            self.fitness_values = np.array([np.std((np.array(fitness) / np.array(self.max_weight)).tolist(), ddof=0) for fitness in knapsack_fitness_values])
            #print(self.fitness_values)
            #print(self.fitness_values)
            elite_parents = self.selection(population, self.fitness_values)
            #print(len(elite_parents))
            new_population = []
            #print(self.mutate(self.crossover(elite_parents[0], elite_parents[1])[0]))
            for i in range(self.elite_num):
                for j in range(i, self.elite_num):
                    parent1, parent2 = elite_parents[i], elite_parents[j]
                    child1, child2 = self.crossover(parent1, parent2)
                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)
                    new_population.extend([child1, child2])
            #print(len(new_population))
            population = population + new_population

            new_knapsack_fitness_values = np.zeros(shape=[len(population), self.knapsack_num])
            for new_p_index in range(len(population)):
                for new_knapsack in range(self.knapsack_num):
                    new_knapsack_fitness_values[new_p_index][new_knapsack] = self.fitness_value(population[new_p_index][new_knapsack], new_knapsack)
            fitness_values = np.array([np.std((np.array(fitness) / np.array(self.max_weight)).tolist(), ddof=0) for fitness in new_knapsack_fitness_values])
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
        plt.title('Genetic Algorithm Optimization Process[{:.5f}]'.format(self.best_fitness))
        plt.show()
