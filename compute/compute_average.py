
import os
import numpy as np
import openpyxl
import pandas as pd

path = 'D:\Projects\pythonProject\EdgeFed-MARL-MEC\compute'
filenames = []
averages = []
ages = []
totals = []
for file in os.listdir(path):
    if os.path.splitext(file)[1] == '.txt':
        filenames.append(file.split('.')[0])
        with open(file, encoding='utf-8') as f:
            rewards = []
            for line in f:
                # print(line)
                string = line.split(':')
                # print(string[-1])
                rewards.append(string[-1].split('\n')[0])
            average = [float(s) for s in rewards[2:9000:3]]
            age = [float(s) for s in rewards[1:9000:3]]
            total = [float(s) for s in rewards[0:9000:3]]
            averages.append(np.mean(average))
            ages.append(np.mean(age))
            totals.append(np.mean(total))
# print(filenames)
# print(averages)

# 输出到Excel
result = {'filenames': filenames,
           'averages': averages,
           'ages': ages,
           'totals':totals}
print(result)

df = pd.DataFrame(result)
df.to_excel('result.xlsx')


