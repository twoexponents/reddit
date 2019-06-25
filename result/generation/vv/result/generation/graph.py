import matplotlib.pyplot as plt
import numpy as np
import os
from pylab import axis

from os.path import dirname, join
current_dir = os.getcwd()
file_path = join(current_dir, 'result\\generation\\content.4.txt')
f = open(file_path, 'r')

print ('hi')

epoch = []; content4 = []
lines = f.readlines()
flag = 0
for line in lines:
    if flag == 1:
        idx = line.index('auc:')
        content4.append(float(line[idx+5:].replace('\n','')))
    if 'epochs:' in line:
        flag = 1
    else:
        flag = 0

f.close()

plt.plot(content4, label="content4")

plt.xlabel('epochs(x10)')
plt.ylabel('AUC')

axis(xmin = 1, ymin = 0.5, ymax = 0.6)

# plt.yticks([i for i in range[0.5, 0.65]])
plt.show()

print ('hi')