import numpy as np
from data import Answer


class Data_factory:
    def __init__(self, data_num: int, data_person: float, func):
        # 原始數據
        a = func
        self.original_data = a.answer()[0]
        self.data_num = data_num
        self.data_person = data_person

    def get_data(self):
        # 計算原始數據的總和
        total_sum = int(sum(self.original_data)) * self.data_person

        # 方法 2: 隨機分配
        random_data = np.random.rand(self.data_num)
        random_data /= random_data.sum()  # 正規化為 1
        random_data *= total_sum  # 調整數據使總和等於原始總和

        # 使用 open() 寫入 txt 檔案，最後一行不換行
        with open("p08_p.txt", "w") as file:
            for i, value in enumerate(random_data):
                if i < len(random_data) - 1:
                    file.write(f"{value}\n")  # 除最後一行外，其他行換行
                else:
                    file.write(f"{value}")    # 最後一行無換行
