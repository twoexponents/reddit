import tensorflow as tf
import torch
import mybertlib
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

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

user_features_fields = ['posts', 'comments']
common_features_fields = ['vader_score', 'vader', 'difficulty']
len_liwc_features = 93
len_bert_features = 768
#len_liwc_features = 29
#input_dim = len(user_features_fields) + len_liwc_features + len_bert_features
input_dim = len_bert_features

output_dim = 2 # (range 0 to 1)
hidden_size = 128
learning_rate = 0.001
batch_size = 128
epochs = 1000
MAX_LEN = 128

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
    #with open('/home/jhlim/data/contentfeatures.others.p', 'rb') as f:
    #    d_features = pickle.load(f)
    with open('/home/jhlim/data/bertfeatures1.p', 'rb') as f:
        d_bertfeatures = pickle.load(f)
    
    print ('features are loaded')

    for seq_length in range(input_length, input_length+1):
        f = open('/home/jhlim/data/seq.learn.%d.csv'%(seq_length), 'r')
        learn_instances = list(map(lambda x:x.replace('\n', '').split(','), f.readlines()))
        f.close() 
        f = open('/home/jhlim/data/seq.test.%d.csv'%(seq_length), 'r')
        test_instances = list(map(lambda x:x.replace('\n', '').split(','), f.readlines()))
        f.close()

        #np.random.shuffle(learn_instances)

        learn_X = []; learn_Y = []
            
        for seq in learn_instances:
            sub_x = []

            try:
                for element in seq[:-1]: # seq[-1] : Y. element: 't3_7dfvv'
                    user_features = []; liwc_features = []; bert_features = []

                    #if element in d_userfeatures and element in d_features:
                    if element in d_bertfeatures:
                        #user_features = d_userfeatures[element]['user'][0:2]
                        #liwc_features = d_features[element]['liwc']
                        bert_features = d_bertfeatures[element]
                    else:
                        continue
                    
                    #if user_features != [] and liwc_features != []:
                    if bert_features != []:
                        #sub_x.append(np.array(user_features + liwc_features + bert_features))
                        sub_x.append(np.array(bert_features))

                if (len(sub_x) == seq_length):
                    learn_X.append(np.array(sub_x)) # feature list
                    learn_Y.append(float(seq[-1]))

            except Exception as e:
                continue

        print ('size of learn_Y: %d' % len(learn_Y))
        print (Counter(learn_Y)) # shows the number of '0' and '1'

        learn_X_reshape = np.reshape(np.array(learn_X), [-1, seq_length*input_dim]) # row num = file's row num
        sample_model = RandomUnderSampler(random_state=42) # random_state = seed. undersampling: diminish majority class
        learn_X, learn_Y = sample_model.fit_sample(learn_X_reshape, learn_Y)
        learn_X = np.reshape(learn_X, [-1, seq_length, input_dim])

        matrix = []
        for v1, v2 in zip(learn_X, learn_Y):
            matrix.append([v1, v2])

        np.random.shuffle(matrix)
        learn_X = list(map(itemgetter(0), matrix))
        learn_Y = list(map(itemgetter(1), matrix))

        print (Counter(learn_Y))


        #np.random.shuffle(test_instances)

        test_X = []; test_Y = []

        for seq in test_instances:
            sub_x = []

            try:
                for element in seq[:-1]:
                    user_features = []; liwc_features = []; bert_features = []
                    
                    #if element in d_userfeatures and element in d_features:
                    if element in d_userfeatures and element in d_bertfeatures:
                        user_features = d_userfeatures[element]['user']
                        #liwc_features = d_features[element]['liwc']
                        bert_features = d_bertfeatures[element]
                    else:
                        continue

                    #if user_features != [] and liwc_features != []:
                    if bert_features != []:
                        if exclude_newbie == 1 and user_features == [0.0, 0.0, 0.0]:
                            continue
                        sub_x.append(np.array(bert_features))

                if (len(sub_x) == seq_length):
                    test_X.append(np.array(sub_x))
                    test_Y.append(float(seq[-1]))

            except Exception as e:
                continue

        print ('size of test_Y: %d' % len(test_Y))
        print (Counter(test_Y))

        print ('Data loading Complete learn:%d, test:%d'%(len(learn_Y), len(test_Y)))
        tf.reset_default_graph()


        # 2. Run RNN
        tf.set_random_seed(50)
        X = tf.placeholder(tf.float32, [None, seq_length, input_dim])
        Y = tf.placeholder(tf.int32, [None])

        is_training = tf.placeholder(tf.bool)

        weights = {}
        biases = {}

        cells = []
        for _ in range(1):
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size,
                                                   state_is_tuple=True,
                                                   activation=tf.nn.relu)
            cells.append(cell)

        cells = tf.nn.rnn_cell.MultiRNNCell(cells) # stackedRNN

        outputs, states = tf.nn.dynamic_rnn(cells, X,
                dtype=tf.float32) # called RNN driver

        outputs = outputs[:, -1]

        bn_output = tf.contrib.layers.batch_norm(outputs, center=True, scale=True, is_training=is_training)

        # three-level MLP
        key = 'fc_l1'
        weights[key] = tf.Variable(tf.random_normal([hidden_size, hidden_size]))
        biases[key] = tf.Variable(tf.random_normal([hidden_size]))

        key = 'fc_l2'
        weights[key] = tf.Variable(tf.random_normal([hidden_size, hidden_size]))
        biases[key] = tf.Variable(tf.random_normal([hidden_size]))

        key = 'fc_l3'
        weights[key] = tf.Variable(tf.random_normal([hidden_size, 2]))
        biases[key] = tf.Variable(tf.random_normal([2]))
        
        optimizers = {}
        pred = []

        # Using dropout -> Fail
        # Using only batch normalization (w/ Relu) -> Fail
        # Using only batch normalization (w/o relu) -> better

        #10_dropout = tf.layers.dropout(outputs, rate=1-keep_prob, training=is_training)

        #l1_output = tf.matmul(bn_output, weights['fc_l1']) + biases['fc_l1']
        l1_output = tf.nn.relu(tf.matmul(bn_output, weights['fc_l1']) + biases['fc_l1']) # might move relu layer to the behind of bn
        l1_bn_output = tf.contrib.layers.batch_norm(l1_output, center=True, scale=True, is_training=is_training)
        #l1_dropout = tf.layers.dropout(l1_output, rate=1-keep_prob, training=is_training)

        #l2_output = tf.matmul(l1_bn_output, weights['fc_l2']) + biases['fc_l2']
        l2_output = tf.nn.relu(tf.matmul(l1_bn_output, weights['fc_l2']) + biases['fc_l2'])
        l2_bn_output = tf.contrib.layers.batch_norm(l2_output, center=True, scale=True, is_training=is_training)

        logits = tf.matmul(l2_bn_output, weights['fc_l3']) + biases['fc_l3']
        labels = tf.one_hot(Y, depth=output_dim)

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
        cost = tf.reduce_mean(loss)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        optimizers = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        optimizers = tf.group([optimizers, update_ops])

        pred.append(tf.argmax(tf.nn.softmax(logits), 1))


        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for e in range(epochs):

                # train batch by batch
                batch_index_start = 0
                batch_index_end = batch_size

                for i in range(int(len(learn_X)/batch_size)):
                    X_train_batch = learn_X[batch_index_start:batch_index_end]
                    Y_train_batch = learn_Y[batch_index_start:batch_index_end]

                    opt, c = sess.run([optimizers, cost],
                            feed_dict={X: X_train_batch, Y: Y_train_batch, is_training:True})
                    

                    batch_index_start += batch_size
                    batch_index_end += batch_size

                if (e % 10 == 0 and e != 0):
                    print ('epochs : %d, cost: %.8f'%(e, c))
                    # TEST
                    rst, c = sess.run([pred, cost], feed_dict={X: test_X, Y: test_Y, is_training:False})

                    out = np.vstack(rst).T

                    #print ('# predict', Counter(out))
                    #print ('# test', Counter(list(map(lambda x:x[0], test_Y))))

                    predicts = []

                    #f = open('../result/result.rnn.%d.tsv'%(seq_length), 'w')
                    for v1, v2 in zip(out, test_Y):
                        #f.write('%d,%s\n'%(v1, v2))
                        decision = False

                        if v1 == int(v2):
                            decision = True
                        predicts.append(decision)

                    print ('seq_length: %d, # predicts: %d, # corrects: %d, acc: %f, auc: %f' %(seq_length, len(predicts), len(list(filter(lambda x:x, predicts))), accuracy_score(test_Y, out), roc_auc_score(test_Y, out)))
                    #print (precision_recall_fscore_support(list(map(int, test_Y)), out))

            print ('\n\n')
    print ('work time: %s sec'%(time.time() - start_time))


if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv])

