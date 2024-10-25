from algorithm import GeneticAlgorithm
from data import Answer
import os

answer = Answer("p07_c.txt", "p07_p.txt", "p07_o.txt")
ga = GeneticAlgorithm(particle=30, Elite_num=40, CrossoverRate=0.9, MutationRate=0.2, MaxIteration=100,
                      values=answer.answer()[0], max_weight=answer.answer()[1], old_value=answer.answer()[2])
best_solution, best_fitness = ga.genetic_algorithm()

os.system('cls' if os.name == 'nt' else 'clear')
print("a = ", best_solution.tolist())
print("Best Fitness:", best_fitness)
ga.show()
