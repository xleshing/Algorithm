from algorithm import GeneticAlgorithm
from data import Get_data
from data_factory import Data_factory
from print_solution import print_solution
import os


def main(knapsack, values, old_values, data_num, data_percent, max_iteration=100, crossover_rate=0.9, mutation_rate=0.1, elite_num=40,
         particle=100):
    get_data = Get_data(knapsack, values, old_values)

    data = Data_factory(data_num=data_num, data_percent=data_percent, func=get_data, txt_file_name=values)

    data.data()

    dim = len(get_data.get_data()[0])

    ga = GeneticAlgorithm(dim=dim, particle=particle, Elite_num=elite_num, CrossoverRate=crossover_rate,
                          MutationRate=mutation_rate, MaxIteration=max_iteration,
                          values=get_data.get_data()[1], max_weight=get_data.get_data()[0], old_value=get_data.get_data()[2])
    best_solution, best_fitness = ga.genetic_algorithm()

    os.system('cls' if os.name == 'nt' else 'clear')
    solution = print_solution(best_solution.tolist(), get_data, dim)
    solution.print_solution()
    print("Best Fitness:", best_fitness)
    ga.show()


main(particle=30, data_num=1000, data_percent=[0.5, 0.3, 0.5], knapsack="knapsack.txt", values="values.txt", old_values="old_values.txt")
