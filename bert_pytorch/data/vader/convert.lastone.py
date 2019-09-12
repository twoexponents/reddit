import numpy as np
import sys
import pickle

with open('/home/jhlim/data/commentbodyfeatures.p', 'rb') as f:
    sentencefile = pickle.load(f)


filename = sys.argv[1]
f = open('/home/jhlim/data/vader/%s'%filename, 'r')
seq_length = int(filename.split('.')[2])
print ('seq_length: ', seq_length)

learn_instances = list(map(lambda x:x.replace('\n', '').split(','), f.readlines()))

filename = filename.replace('csv', 'tsv')
f = open(filename, 'w')
for instance in learn_instances:
    id = instance[0]
    label = instance[seq_length]
    element = instance[seq_length-1]
    sentence = ""
    if element in sentencefile:
        sentence = sentencefile[element].replace('\n', ' ')
    if len(sentence) < 1:
        continue
    if len(sentence) > 100:
        continue

    f.write(element + '\t' + label + '\t' + '*' + '\t' + sentence + '\n')

    

