import numpy as np
import random
import matplotlib.pyplot as plt

class GeneticAlgorithm:
    def __init__(self, Number=100, Elite_num=40, CrossoverRate=0.9, MutationRate=0.1,
                 MaxIteration=100, values=None, weight=None, max_weight=None):
        self.values = values
        self.weights = weight
        self.max_weight = max_weight
        self.N = Number
        self.D = len(self.values)
        self.n = Elite_num
        self.cr = CrossoverRate
        self.mr = MutationRate
        self.max_iter = MaxIteration
        self.best_solution = None
        self.best_fitness = -1
        self.fitness_values = np.array([])
        self.best_fitness_list = []

    def fitness_value(self, solution):
        total_value = np.sum(np.array(self.values) * np.array(solution))
        total_weight = np.sum(np.array(self.weights) * np.array(solution))
        if total_weight > self.max_weight:
            return 0
        else:
            return total_value

    def init_population(self):
        return [np.random.randint(2, size=self.D) for _ in range(self.N)]

    def selection(self, population, fitness_values):
        elite_index = np.argsort(fitness_values)[-self.n:]
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
        for i in range(len(child)):
            if random.random() < self.mr:
                child[i] = 1 - child[i]
        return child

    def genetic_algorithm(self):
        population = self.init_population()
        for _ in range(self.max_iter):
            self.fitness_values = np.array([self.fitness_value(chromosome) for chromosome in population])
            elite_parents = self.selection(population, self.fitness_values)
            new_population = []
            for i in range(self.n):
                for j in range(i, self.n):
                    parent1, parent2 = elite_parents[i], elite_parents[j]
                    child1, child2 = self.crossover(parent1, parent2)
                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)
                    new_population.extend([child1, child2])
            population = population + new_population
            fitness_values = np.array([self.fitness_value(chromosome) for chromosome in population])
            idx_to_keep = np.argsort(fitness_values)[-self.N:]
            population = [population[idx] for idx in idx_to_keep]
            self.best_fitness_list.append(np.max(self.fitness_values))
            if np.max(self.fitness_values) > self.best_fitness:
                self.best_solution = population[np.argmax(self.fitness_values)]
                self.best_fitness = np.max(self.fitness_values)
        return self.best_solution, self.best_fitness

    def show(self):
        plt.plot(self.best_fitness_list)
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.title('Genetic Algorithm Optimization Process')
        plt.show()