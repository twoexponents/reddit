from __future__ import division
import tensorflow as tf
import numpy as np
import sys
import pickle
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from collections import Counter
from operator import itemgetter
from mytensorlib import runRNNModel

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


user_features_fields = ['posts', 'comments']

input_dim = len(user_features_fields)

output_dim = 1 # (range 0 to 1)
hidden_size = 5
learning_rate = 0.005
batch_size = 32
epochs = 110
keep_rate = 0.5
exclude_newbie = 0
print_file = 1

def main(argv):
    start_time = time.time()
    exclude_newbie = 0; input_length = 1
    if len(sys.argv) >= 3:
        exclude_newbie = int(sys.argv[2])
    if len(sys.argv) >= 2:
        input_length = int(sys.argv[1])
    print ('sequence len: %d' % (input_length))
    print ('learning_rate: %f, batch_size %d, epochs %d' % (learning_rate, batch_size, epochs))
    print ('exclude_newbie: %d'%(exclude_newbie))

    # 1.1 load feature dataset
    with open('/home/jhlim/data/userfeatures.activity.p', 'rb') as f:
        d_userfeatures = pickle.load(f)
    sentencefile = pickle.load(open('/home/jhlim/data/commentbodyfeatures.p', 'rb'))

    print ('features are loaded')

    fa = open('length1commentbody.txt', 'w')
    for seq_length in range(input_length, input_length+1):
        f = open('/home/jhlim/data/seq.learn.%d.csv'%(seq_length), 'r')
        learn_instances = list(map(lambda x:x.replace('\n', '').split(','), f.readlines()))
        f.close()

        #np.random.shuffle(learn_instances)

        body_list = []

        learn_X = []; learn_Y = []
        for seq in learn_instances:
            sub_x = []
            flag = -1;

            try:
                for element in seq[:-1]: # seq[-1] : Y. element: 't3_7dfvv'
                    flag += 1;
                    user_features = []
                    if element in d_userfeatures:
                        user_features = d_userfeatures[element]['user']
                        if flag == 1:
                            if element in sentencefile:
                                body_list.append(sentencefile[element])
                            else:
                                body_list.append('NULL')
                    else:
                        continue
                    
                    if user_features != []:
                        sub_x.append(np.array(user_features))

                if (len(sub_x) == seq_length):
                    learn_X.append(np.array(sub_x)) # feature list
                    learn_Y.append(float(seq[-1]))

            except Exception as e:
                continue

        print ('size of learn_Y: %d' % len(learn_Y))
        print (Counter(learn_Y)) # shows the number of '0' and '1'

        learn_X_reshape = np.reshape(np.array(learn_X), [-1, seq_length*input_dim]) # row num = file's row num
        sample_model = RandomUnderSampler(random_state=42) # random_state = seed. undersampling: diminish majority class
        print (len(learn_X_reshape), len(learn_Y))
        learn_X, learn_Y = sample_model.fit_sample(learn_X_reshape, learn_Y)
        learn_X = np.reshape(learn_X, [-1, seq_length, input_dim])

        matrix = []
        for v1, v2 in zip(learn_X, learn_Y):
            matrix.append([v1, v2])

        np.random.shuffle(matrix)
        learn_X = list(map(itemgetter(0), matrix))
        learn_Y = list(map(lambda x:[x], map(itemgetter(1), matrix)))

        print (Counter(list(map(lambda x:x[0], learn_Y))))

        f = open('/home/jhlim/data/seq.test.%d.csv'%(seq_length), 'r')
        test_instances = list(map(lambda x:x.replace('\n', '').split(','), f.readlines()))
        f.close()

        #np.random.shuffle(test_instances)

        test_X = []; test_Y = []

        for seq in test_instances:
            sub_x = []
            flag = -1

            try:
                for element in seq[:-1]:
                    flag += 1
                    user_features = [] 
                    if element in d_userfeatures:
                        user_features = d_userfeatures[element]['user']
                        if flag == 1:
                            if element in sentencefile:
                                body_list.append(sentencefile[element])
                            else:
                                body_list.append('NULL')
                    else:
                        continue

                    if user_features != []:
                        if exclude_newbie == 1 and user_features == [0.0, 0.0]:
                                continue
                        sub_x.append(np.array(user_features))

                if (len(sub_x) == seq_length):
                    test_X.append(np.array(sub_x))
                    test_Y.append(float(seq[-1]))

            except Exception as e:
                continue
        
        test_Y = list(map(lambda x:[x], test_Y))
        print ('size of test_Y: %d' % len(test_Y))
        print (Counter(list(map(lambda x:x[0], test_Y))))
        
        print ('Data loading Complete learn:%d, test:%d'%(len(learn_Y), len(test_Y)))
        print (body_list)
        body_list.sort(key=len)
        for item in body_list:
            fa.write(item + '\n')


        # 2. Run RNN

        #runRNNModel(seq_length, learn_X, learn_Y, test_X, test_Y, input_dim, hidden_size, learning_rate, batch_size, epochs, keep_rate)


if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv])

