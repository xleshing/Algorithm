from algorithm import PSO
from func import Func
from data import Answer
from test import Test

def main(c, p, o, num_particle=100, max_iter=100):
    a = Answer(c, p, o)
    knapsack = a.answer()[0]
    item = a.answer()[1]
    old_value = a.answer()[2]

    func = Func(knapsack, item, old_value)

    x_max = len(knapsack) ** len(item) - 1
    x_min = 0

    optimizer = PSO(funct=func.fitness_value, num_particle=num_particle, max_iter=max_iter, x_max=x_max, x_min=x_min, knapsack=knapsack)
    sol = optimizer.update()
    test = Test(func.change_sol(sol).tolist(), a)
    test.print_answer()
    optimizer.plot_curve()

main(c="p08_c.txt", p="p08_p.txt", o="p08_o.txt")
