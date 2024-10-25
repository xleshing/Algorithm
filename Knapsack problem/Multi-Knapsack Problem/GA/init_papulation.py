import numpy as np

class Init_population:
    def __init__(self, values: list, max_weight: list, knapsack_num: int, old_value: list, particle_num: int):
        self.values = values
        self.max_weight = max_weight
        self.knapsack_num = knapsack_num
        self.old_value = old_value
        self.item_num = len(self.values)
        self.particle_num = particle_num

    def init_population(self, particle_num: int) -> list:
        if int(particle_num) == 0:
            return []
        return [np.random.randint(self.knapsack_num, size=[self.item_num]) for _ in range(particle_num)]

    def init_population_dim(self, population: list, particle_num: int) -> list:
        if int(particle_num) == 0:
            return []
        sol = [np.zeros(shape=[self.knapsack_num, self.item_num]) for _ in range(particle_num)]
        for chromosome in range(len(population)):
            item = 0
            for knapsack in population[chromosome]:
                sol[chromosome][knapsack, item] = 1
                item += 1
        return sol

    def fitness_value(self, solution: list, knapsack: int, old_value: list):
        total_value = np.sum(np.array(self.values) * np.array(solution)) + old_value[knapsack]
        return total_value

    def fix_population(self, population: list) -> list:
        while True:
            is_remove = False
            index = 0
            while True:
                if index >= len(population):
                    break
                for each_knapsack in range(len(population[index])):
                    if self.fitness_value(population[index][each_knapsack], each_knapsack, self.old_value) > self.max_weight[each_knapsack]:
                        del population[index]
                        is_remove = True
                        break
                    else:
                        is_remove = False

                if not is_remove:
                    index += 1

            new_sol = self.init_population_dim(self.init_population(self.particle_num - len(population)), self.particle_num - len(population))

            population = population + new_sol

            if len(new_sol) == 0:
                break

        return population




