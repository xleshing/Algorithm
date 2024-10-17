import numpy as np

processing_times = np.array([
    [54, 34, 61, 2],
    [9, 15, 89, 70],
    [38, 19, 28, 87],
    [95, 34, 7, 29]])

machines = np.array([
    [2, 0, 3, 1],
    [3, 0, 1, 2],
    [0, 1, 2, 3],
    [0, 2, 1, 3]])

class func:
    def __init__(self, machine_num, job_num, processing_times):
        self.machine_num = machine_num
        self.job_num = job_num
        self.processing_times = processing_times

        self.ans_list = []

        self.sum_len = np.zeros(shape=[self.machine_num, 1])
        self.job_len = np.zeros(shape=[self.job_num, 1])

    def check(self, sol):
        self.sum_len = np.zeros(shape=[self.machine_num, 1])
        self.job_len = np.zeros(shape=[self.job_num, 1])
        self.ans_list = []
        self.ans_list = sol
        for i in range(self.job_num):
            for j in range(self.machine_num):
                machine_time = self.sum_len[j, 0]
                job_time = self.job_len[int(self.ans_list[j, i]), 0]
                proc_time = self.processing_times[int(self.ans_list[j, i]), j]

                new_time = max(machine_time, job_time) + proc_time

                self.sum_len[j, 0] = new_time
                self.job_len[int(self.ans_list[j, i]), 0] = new_time

        return max(self.sum_len)

if __name__ == "__main__":
    f = func(4, 4, processing_times)
    print(f.check(machines))
