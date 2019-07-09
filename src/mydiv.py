# mydev.py

import numpy as np


def stats(p_cont, p_end, l_cont, l_end):
    print 'mean_predict_cont: %.2f, stdev: %.2f'%(div(sum(p_cont), len(p_cont)), np.std(np.array(p_cont), ddof=1))
    print 'mean_predict_end: %.2f, stdev: %.2f'%(div(sum(p_end), len(p_end)), np.std(np.array(p_end), ddof=1))
    print 'mean_label_cont: %.2f, stdev: %.2f'%(div(sum(l_cont), len(l_cont)), np.std(np.array(l_cont), ddof=1))
    print 'mean_label_end: %.2f, stdev: %.2f'%(div(sum(l_end), len(l_end)), np.std(np.array(l_end), ddof=1))
    

def div(a, b):
    if b == 0:
        print 'division by zero'
        return -1
    else:
        return a/float(b)

