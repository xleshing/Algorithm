import numpy as np
import random
import matplotlib.pyplot as plt
from score import func

class GeneticAlgorithm:
    def __init__(self, particle_num=100, Elite_num=40, MaxIteration=1000, machine_num=None, job_num=None, processing_times=None, CrossoverRate=0.7, MutationRate=0.4):
        self.job_num = job_num
        self.machine_num = machine_num
        self.processing_times = processing_times
        self.particle_num = particle_num
        self.elite_num = Elite_num
        self.cr = CrossoverRate
        self.mr = MutationRate
        self.max_iter = MaxIteration
        self.best_solution = None
        self.best_fitness = 1000000
        self.fitness_values = np.array([])
        self.best_fitness_list = []
        self.f = func(self.machine_num, self.job_num, self.processing_times)

    def init_population(self):
        # print([np.array([np.random.permutation(np.arange(self.job_num)) for _ in range(self.machine_num)]) for _ in range(self.particle_num)])
        return [np.array([np.random.permutation(np.arange(self.job_num)) for _ in range(self.machine_num)]) for _ in range(self.particle_num)]

    def selection(self, population, fitness_values):
        elite_index = np.argsort(fitness_values)[:self.elite_num]
        elite_parent = [population[idx] for idx in elite_index]
        return elite_parent

    @staticmethod
    def crossover(parent1, parent2):
        num_crossover_points = random.randint(1, len(parent1) - 1)
        crossover_points = sorted(random.sample(range(1, len(parent1)), num_crossover_points))

        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)

        start = 0
        for i, point in enumerate(crossover_points):
            if i % 2 == 0:
                child1[start:point] = parent1[start:point]
                child2[start:point] = parent2[start:point]
            else:
                child1[start:point] = parent2[start:point]
                child2[start:point] = parent1[start:point]
            start = point

        child1[start:] = parent1[start:]
        child2[start:] = parent2[start:]

        return child1, child2

    def mutate(self, child):
        for i in range(len(child)):  # 遍歷每台機器的工作順序
            if random.random() < self.mr:  # 如果隨機數小於變異率，則執行變異
                # 隨機選擇兩個不同的任務位置進行交換
                idx1, idx2 = random.sample(range(len(child[i])), 2)
                # 交換這兩個位置的任務
                child[i][idx1], child[i][idx2] = child[i][idx2], child[i][idx1]
        return child

    def genetic_algorithm(self):
        population = self.init_population()
        for _ in range(self.max_iter):
            self.fitness_values = np.array([self.f.check(chromosome) for chromosome in population]).flatten()
            elite_parents = self.selection(population, self.fitness_values)
            new_population = []
            for i in range(self.elite_num):
                for j in range(i, self.elite_num):
                    parent1, parent2 = elite_parents[i], elite_parents[j]
                    child1 = []
                    child2 = []
                    for k in range(self.machine_num):
                        child_temp1, child_temp2 = self.crossover(parent1[k, :], parent2[k, :])
                        child1.append(child_temp1)
                        child2.append(child_temp2)
                    child1 = np.array(child1)
                    child2 = np.array(child2)
                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)
                    new_population.extend([child1, child2])
            population = population + new_population
            fitness_values = np.array([self.f.check(chromosome) for chromosome in population])
            idx_to_keep = np.argsort(fitness_values)[:self.particle_num]
            population = [population[int(idx)] for idx in idx_to_keep]
            self.best_fitness_list.append(np.min(self.fitness_values))
            if np.min(self.fitness_values) < self.best_fitness:
                self.best_solution = population[np.argmin(self.fitness_values)]
                self.best_fitness = np.min(self.fitness_values)
            print(self.best_fitness)
        return self.best_solution, self.best_fitness

    def show(self):
        plt.plot(self.best_fitness_list)
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.title('Genetic Algorithm Optimization Process')
        plt.show()

if __name__ == "__main__":
    ga = GeneticAlgorithm()
    print(ga.selection(ga.init_population(), [4, 2, 3]))
