from algorithm import GeneticAlgorithm
import numpy as np



processing_times = np.array([
    [54, 34, 61, 2],
    [9, 15, 89, 70],
    [38, 19, 28, 87],
    [95, 34, 7, 29]])
ga = GeneticAlgorithm(job_num=4, machine_num=4, processing_times=processing_times, MutationRate=0.6, Elite_num=10)
best_solution, best_fitness = ga.genetic_algorithm()

print(best_solution, best_fitness)
ga.show()


