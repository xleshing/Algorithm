import numpy as np
from curve_func import Curve
# 假設初始價格和初始購買量
base_price, base_demand = 1557.504 , 631325.92644
base_sales = base_price * base_demand  # 基礎銷售額
best_price = base_price
best_demand = base_demand

# 給定價格和購買量增長曲線
curve = np.array([[0.3, -0.01],
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

# 基礎的貪婪算法來找出最大銷售額
def find_max_sales(base_price, base_demand, curve):
    fun = Curve(base_price, base_demand, 4, 2)
    max_sales = 0
    best_combination = None

    for case in range(len(curve)):
        # 計算新的價格和需求量
        (new_price, new_demand) = fun.result(case)
        s = fun.salse(case)
        l = fun.labors(case)
        sales = fun.ans(case)
        print(sales, s,l)

        # 比較銷售額並記錄最大值
        if sales > max_sales:
            max_sales = sales
            best_combination = (curve[case][0], curve[case][1])
            best_price = new_price
            best_demand = new_demand

    return max_sales, best_combination, best_price, best_demand

# 使用算法
max_sales, best_combination, final_price, final_demand = find_max_sales(base_price, base_demand, curve)
print("最大營業額：", max_sales)
print("最佳組合（價格增長值，購買量增長值）：", best_combination)
print(final_price, ",", final_demand)
