import numpy as np
import sys
import pickle

with open('/home/jhlim/data/commentbodyfeatures.p', 'rb') as f:
    sentencefile = pickle.load(f)

print ('Start converting')
for seq_length in range(1, 10):
    files = ['seq.learn.%d.csv'%(seq_length), 'seq.test.%d.csv'%(seq_length)]

    for filename in files:
        f = open('/home/jhlim/data/%s'%filename, 'r')
        seq_length = int(filename.split('.')[2])
        print ('seq_length: ', seq_length)

        learn_instances = list(map(lambda x:x.replace('\n', '').split(','), f.readlines()))

        filename = filename.replace('csv', 'tsv')
        f = open(filename, 'w')
        for instance in learn_instances:
            id = instance[0]
            label = instance[seq_length]
            sub_x = []
            for element in instance[:-1]:
                if element in sentencefile:
                    sentence = sentencefile[element].replace('\n', ' ')
                    if len(sentence) < 1:
                        continue
                    sub_x.append(np.array(sentence))
            if len(sub_x) != seq_length:
                continue
            sentence = str(sub_x[0])
            flag = False;
            for element in sub_x[1:]:
                if len(str(element)) > 200:
                    flag = True;
                    break;
                sentence = sentence + ". " + str(element)

            if not flag:
                f.write(id + '\t' + label + '\t' + '*' + '\t' + sentence + '\n')

    

