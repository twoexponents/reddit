import re
import numpy as np
import matplotlib.pyplot as plt
import operator
import math

NUM_SIZE = 8

# w/ PFC
f = open('len_distribution.txt', 'r')
lines = f.readlines()
lst = []
for line in lines:
    items = line.split(',')
    for i in range(int(items[1])):
        lst.append(int(items[0]))
        
p2 = 1. * np.arange(len(lst))/(len(lst) - 1)
plt.rcParams.update({'font.size': 22})
plt.plot(lst, p2)

#plt.xscale('log')
plt.rcParams['axes.grid'] = True
plt.xlabel('Depth of Thread')
plt.ylabel('CDF')
#plt.legend(loc='upper right')
plt.show()


