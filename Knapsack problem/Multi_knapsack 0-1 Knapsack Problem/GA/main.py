from algorithm import GeneticAlgorithm
from data import Answer
import numpy as np


class Main:
    def __init__(self):
        self.answer = Answer("p07_c.txt", "p07_p.txt", "p07_w.txt", "p07_s.txt")
        self.ga = GeneticAlgorithm(values=self.answer.answer()[0], weight=self.answer.answer()[1],
                                   max_weight=self.answer.answer()[2])
        self.best_solution, self.best_fitness = self.ga.genetic_algorithm()

    def main(self):
        print("Best Solution:", self.best_solution, "Best Solution:", self.answer.answer()[3])
        print("Best Fitness:", self.best_fitness, "Best Fitness:", self.answer.answer()[4])
        if np.all(self.best_solution == self.answer.answer()[3]):
            print("Correct")
        else:
            print("Incorrect")
        self.ga.show()


if __name__ == "__main__":
    main = Main()
    main.main()
