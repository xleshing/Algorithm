import numpy as np


class Curve:
    def __init__(self, price, demand, year, item):
        self.item = item
        self.year = year
        self.price = price
        self.demand = demand
        self.curve = np.array([[0.3, -0.01],
                               [0.25, 0],
                               [0.2, 0.01],
                               [0.15, 0.02],
                               [0.1, 0.03],
                               [0.05, 0.06],
                               [0, 0.15],
                               [-0.05, 0.42],
                               [-0.1, 1.23],
                               [-0.15, 2.23],
                               [-0.2, 3.23]])

    def result(self, case) -> tuple[float, float]:
        return self.price + float(self.price * self.curve[case][0]), self.demand + float(
            self.demand * self.curve[case][1])

    def salse(self, case) -> float:
        return (self.price + self.price * self.curve[case][0]) * (
                self.demand + self.demand * self.curve[case][1])

    def labors(self, case) -> float:
        labor_curve = np.array([
            [17.6, 19.2 * 1.02 ** 0],
            [17.6, 19.2 * 1.02 ** 1],
            [17.6, 19.2 * 1.02 ** 2],
            [17.6, 19.2 * 1.02 ** 3],
            [17.6, 19.2 * 1.02 ** 4]
        ])
        time_curve = np.array([
            [5, 5],
            [5, 10],
            [5, 20]
        ])

        return (self.result(case)[1] *
                (float(labor_curve[self.year][0] * time_curve[self.item][0]) +
                 float(labor_curve[self.year][1] * time_curve[self.item][1])))

    def ans(self, case) -> float:
        return float(self.salse(case) * 0.6 - self.labors(case))


if __name__ == "__main__":
    f = Curve(832.0, 180908.64, 2, 1)
    print(f.result(10))
    print(f.salse(10))
    print(f.labors(10))
    print(f.ans(10))
