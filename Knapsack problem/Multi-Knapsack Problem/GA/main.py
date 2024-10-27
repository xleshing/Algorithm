from algorithm import GeneticAlgorithm
from data import Answer
import os

answer = Answer("p08_c.txt", "p08_p.txt", "p08_o.txt")
ga = GeneticAlgorithm(particle=30, Elite_num=40, CrossoverRate=0.9, MutationRate=0.1, MaxIteration=100,
                      values=answer.answer()[0], max_weight=answer.answer()[1], old_value=answer.answer()[2])
best_solution, best_fitness = ga.genetic_algorithm()

os.system('cls' if os.name == 'nt' else 'clear')
print("a = ", best_solution.tolist())
print("Best Fitness:", best_fitness)
ga.show()
