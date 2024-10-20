from algorithm import GeneticAlgorithm
from data import Answer


class Main:
    def __init__(self):
        self.answer = Answer("p07_c.txt", "p07_p.txt")
        self.ga = GeneticAlgorithm(MaxIteration=100, values=self.answer.answer()[0],
                                   max_weight=self.answer.answer()[1])
        self.best_solution, self.best_fitness = self.ga.genetic_algorithm()

    def main(self):
        print("Best Solution:", self.best_solution.tolist())
        print("Best Fitness:", self.best_fitness)
        self.ga.show()


if __name__ == "__main__":
    main = Main()
    main.main()
