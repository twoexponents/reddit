#import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np

from operator import itemgetter

def draw(data):
    plt.rcParams.update({'font.size': 22})
    data = data[1:]
    data = list(map(lambda x:x/1000, data))
    
    plt.plot(data)

    plt.xlabel('Length of Sequence')
    plt.ylabel('Number of Sequences (x1000)')
    plt.rcParams["figure.figsize"] = (8, 4)
    plt.rcParams['axes.grid'] = True
    plt.show()
    
def main():
    f = open('/home/jhlim/SequencePrediction/data/sequences.csv')
    lines = f.readlines()

    d = {}
    for line in lines:
        lst = line.split(',')
        length = len(lst)
        if length not in d:
            d[length] = 0
        d[length] += 1


    s_list = sorted(d.items(), key=itemgetter(0))
    draw(list(map(lambda x:x[1], s_list)))

    f.close()

if __name__ == '__main__':
    main()

    

