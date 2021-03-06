import numpy as np
import sys
import pickle

input_dim = 1

with open('/home/jhlim/data/commentbodyfeatures.p', 'rb') as f:
    sentencefile = pickle.load(f)

print ('Start converting')
for seq_length in range(10, 11):
    files = ['seq.learn.less%d.csv'%(seq_length), 'seq.test.less%d.csv'%(seq_length)]

    for filename in files:
        f = open('/home/jhlim/data/%s'%filename, 'r')
        seq_length = int(filename.split('.')[2][4:6])
        print ('seq_length: ', seq_length)

        learn_instances = list(map(lambda x:x.replace('\n', '').split(','), f.readlines()))

        filename = filename.replace('csv', 'tsv')
        f = open(filename, 'w')
        for instance in learn_instances:
            label = instance[-1]
            element = instance[-2]
            sentence = ""
            if element in sentencefile:
                sentence = sentencefile[element].replace('\n', ' ').replace('\t', ' ').replace('"', '')
            if len(sentence) < 1 or len(sentence) > 500:
                continue

            f.write(element + '\t' + label + '\t' + '*' + '\t' + sentence + '\n')

