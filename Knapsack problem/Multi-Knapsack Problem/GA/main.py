from algorithm import GeneticAlgorithm
from data import Answer
from data_factory import Data_factory
from test import Test
import os

def main(c, p, o, data_num, data_person, max_iteration=100, crossover_rate=0.9, mutation_rate=0.1, elite_num=40, particle=30):
    answer = Answer(c, p, o)
    data = Data_factory(data_num=data_num, data_person=data_person, func=answer)

    data.get_data()

    ga = GeneticAlgorithm(particle=particle, Elite_num=elite_num, CrossoverRate=crossover_rate, MutationRate=mutation_rate, MaxIteration=max_iteration,
                          values=answer.answer()[1], max_weight=answer.answer()[0], old_value=answer.answer()[2])
    best_solution, best_fitness = ga.genetic_algorithm()

    os.system('cls' if os.name == 'nt' else 'clear')
    test = Test(best_solution.tolist(), answer)
    test.print_answer()
    print("Best Fitness:", best_fitness)
    ga.show()

main(data_num=100, data_person=0.35, c="p08_c.txt", p="p08_p.txt", o="p08_o.txt")
