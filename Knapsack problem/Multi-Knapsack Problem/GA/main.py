from algorithm import GeneticAlgorithm
from data import Answer
import os

answer = Answer("p07_c.txt", "p07_p.txt")
ga = GeneticAlgorithm(particle=1000, Elite_num=40, CrossoverRate=0.9, MutationRate=0.2, MaxIteration=100,
                      values=answer.answer()[0],
                      max_weight=answer.answer()[1])
best_solution, best_fitness = ga.genetic_algorithm()

os.system('cls' if os.name == 'nt' else 'clear')
print("Best Solution:", "\n", best_solution)
print("Best Fitness:", best_fitness)
ga.show()
