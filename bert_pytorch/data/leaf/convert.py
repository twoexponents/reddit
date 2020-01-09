import numpy as np
import sys
import pickle

with open('/home/jhlim/data/commentbodyfeatures.p', 'rb') as f:
    sentencefile = pickle.load(f)

print ('Start converting')
for seq_length in range(1, 6):
    files = ['seq.learn.%d.csv'%(seq_length), 'seq.test.%d.csv'%(seq_length)]

    for filename in files:
        f = open('/home/jhlim/data/%s'%filename, 'r')
        seq_length = int(filename.split('.')[2])
        print ('seq_length: ', seq_length)

        learn_instances = list(map(lambda x:x.replace('\n', '').split(','), f.readlines()))

        filename = filename.replace('csv', 'tsv')
        filename = filename.replace('seq', 'look')
        f = open(filename, 'w')
        for instance in learn_instances:
            id = '\t'.join(instance[:-1])
            label = instance[-1]
            sub_x = []
            for i, element in enumerate(instance[:-1]):
                if element in sentencefile:
                    sentence = sentencefile[element].replace('\n', ' ')
                    sentence = sentence.replace('\t', ' ')
                    #if len(sentence) < 1 or len(sentence) > 500:
                    #    break
                    sub_x.append('(' + str(i+1) + ') ' + sentence)

            if len(sub_x) != seq_length:
                continue

            sentences = '\n'.join(sub_x) + '\n' + instance[-1]

            f.write(id + '\n' + sentences + '\n')
            
            #f.write(id + '\t' + label + '\t' + '*' + '\t' + sentences + '\n')

    

