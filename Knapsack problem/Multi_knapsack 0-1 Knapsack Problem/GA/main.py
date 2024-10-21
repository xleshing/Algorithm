from algorithm import GeneticAlgorithm
from data import Answer
import os


class Main:
    def __init__(self):
        self.answer = Answer("p07_c.txt", "p07_p.txt")
        self.ga = GeneticAlgorithm(particle=1000, Elite_num=40, CrossoverRate=0.9, MutationRate=0.2, MaxIteration=100, values=self.answer.answer()[0],
                                   max_weight=self.answer.answer()[1])
        self.best_solution, self.best_fitness = self.ga.genetic_algorithm()

    def main(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print("Best Solution:", "\n", self.best_solution)
        print("Best Fitness:", self.best_fitness)
        self.ga.show()


if __name__ == "__main__":
    main = Main()
    main.main()
