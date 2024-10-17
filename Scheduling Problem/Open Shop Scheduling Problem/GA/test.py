import time
import os

# 定義np的數據
np_data = [
    [[1, 2], [3, 4], [5, 6], [7, 8]],
    [[11, 21], [13, 14], [15, 16], [17, 18]],
    [[12, 22], [32, 42], [25, 62], [72, 82]],
    [[13, 23], [33, 34], [53, 63], [37, 38]],
    [[41, 24], [34, 44], [54, 64], [74, 84]]
]

# 循環輪流打印每個np並刷新畫面
def print_np():
    while True:
        for np in np_data:
            os.system('cls' if os.name == 'nt' else 'clear')  # 清屏
            for row in np:
                print(f"{row[0]},{row[1]}")
            time.sleep(0.5)  # 延遲0.5秒

# 執行函數
print_np()