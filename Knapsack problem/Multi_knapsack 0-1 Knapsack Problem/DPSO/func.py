import numpy as np
import math


class func:
    def __init__(self, machine_num, job_num, processing_times):
        self.machine_num = machine_num
        self.job_num = job_num
        self.processing_times = processing_times

        self.ans_list = []

        self.sum_len = np.zeros(shape=[self.machine_num, 1])
        self.job_len = np.zeros(shape=[self.job_num, 1])

    def get_permutation(self, k):
        n = self.job_num
        numbers = list(range(n))
        permutation = []

        while n > 0:
            n -= 1
            fact = math.factorial(n)
            index = k // fact
            permutation.append(int(numbers.pop(index)))
            k %= fact

        return permutation

    def check(self, sol):
        self.sum_len = np.zeros(shape=[self.machine_num, 1])
        self.job_len = np.zeros(shape=[self.job_num, 1])
        self.ans_list = []

        for k in sol:
            self.ans_list.append(self.get_permutation(k))

        self.ans_list = np.array(self.ans_list).reshape(self.machine_num, self.job_num)

        if np.any(np.unique(self.ans_list, return_counts=True)[1] <= 4):
            for i in range(self.job_num):
                for j in range(self.machine_num):
                    machine_time = self.sum_len[j, 0]
                    job_time = self.job_len[int(self.ans_list[j, i]), 0]

                    proc_time = self.processing_times[j, int(self.ans_list[j, i])]

                    new_time = max(machine_time, job_time) + proc_time

                    self.sum_len[j, 0] = new_time
                    self.job_len[int(self.ans_list[j, i]), 0] = new_time

            return max(self.sum_len)
        else:
            return max(self.sum_len)*10000
