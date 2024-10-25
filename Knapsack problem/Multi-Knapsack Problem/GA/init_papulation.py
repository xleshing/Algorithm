import numpy as np

class Initial_population:
    def __init__(self, values, max_weight, knapsack_num, old_value, particle_num):
        self.values = values
        self.max_weight = max_weight
        self.knapsack_num = knapsack_num
        self.old_value = old_value
        self.item_num = len(self.values)
        self.particle_num = particle_num

    def init_population(self, particle_num):
        if int(particle_num) == 0:
            return np.array([])
        return [np.random.randint(self.knapsack_num, size=[self.item_num]) for _ in range(particle_num)]

    def init_population_dim(self, population, particle_num):
        if int(particle_num) == 0:
            return np.array([])
        sol = [np.zeros(shape=[self.knapsack_num, self.item_num]) for _ in range(particle_num)]
        for chromosome in range(len(population)):
            item = 0
            for knapsack in population[chromosome]:
                sol[chromosome][knapsack, item] = 1
                item += 1
        return sol

    def fitness_value(self, solution, knapsack, old_value):
        total_value = np.sum(np.array(self.values) * np.array(solution)) + old_value[knapsack]
        return total_value

    def fix_population(self, population):
        population = np.array(population)
        while True:
            is_remove = False
            index = 0
            while index < len(population):
                for each_knapsack in range(len(population[index])):
                    if self.fitness_value(population[index][each_knapsack], each_knapsack, self.old_value) > self.max_weight[each_knapsack]:
                        population = np.delete(population, index, 0)
                        is_remove = True
                        break
                    else:
                        is_remove = False

                if not is_remove:
                    index += 1

            new_sol = self.init_population_dim(self.init_population(self.particle_num - len(population)), self.particle_num - len(population))
            # print(new_sol)
            print(new_sol, "new_sol")

            if len(new_sol) == 0:
                break

            population = population + new_sol

        return population




