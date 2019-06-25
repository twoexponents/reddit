import matplotlib.pyplot as plt
import numpy as np
from pylab import axis

f = open('./content.4.txt', 'r')

lines = f.readlines()
flag = 0
for line in lines:
    if flag == 1:
        idx = line.index('auc:')
        print (idx)
        # str = 
    if 'epochs:' in line:
        flag = 1
    else:
        flag = 0


f.close()