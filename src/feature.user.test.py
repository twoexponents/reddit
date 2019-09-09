from __future__ import division
import tensorflow as tf
import numpy as np
import sys
import pickle
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from collections import Counter
from operator import itemgetter

#user_features_fields = ['posts', 'comments', 'convs', 'entropy_conv']
user_features_fields = ['posts', 'comments']
#user_features_fields = ['posts']
#user_features_fields = ['comments']

input_dim = len(user_features_fields)

output_dim = 1 # (range 0 to 1)
hidden_size = 100
learning_rate = 0.005
batch_size = 100
epochs = 500

def main(argv):
    start_time = time.time()
    if len(sys.argv) <= 1:
        print "sequence len: 1"
        input_length = 1
    else:
        input_length = int(sys.argv[1])

    print 'learning_rate: %f, batch_size %d, epochs %d' %(learning_rate, batch_size, epochs)

    # 1.1 load feature dataset
    d_userfeatures = pickle.load(open('../data/userfeatures.activity.p', 'r'))

    print 'features are loaded'

    for seq_length in range(input_length, input_length+1):
        f = open('../data/seq.learn.%d.csv'%(seq_length), 'r')
        learn_instances = map(lambda x:x.replace('\n', '').split(','), f.readlines())
        f.close()

        np.random.shuffle(learn_instances)

        learn_X = []; learn_Y = []
        for seq in learn_instances:
            sub_x = []

            try:
                for element in seq[:-1]: # seq[-1] : Y. element: 't3_7dfvv'
                    user_features = []
                    if d_userfeatures.has_key(element):
                        user_features = d_userfeatures[element]['user'][0:2]
                    else:
                        continue
                    
                    if user_features != []:
                        sub_x.append(np.array(user_features))

                if (len(sub_x) == seq_length):
                    learn_X.append(np.array(sub_x)) # feature list
                    learn_Y.append(float(seq[-1]))

            except Exception as e:
                continue

        print 'size of learn_Y: %d' % len(learn_Y)
        print Counter(learn_Y) # shows the number of '0' and '1'

        learn_X_reshape = np.reshape(np.array(learn_X), [-1, seq_length*input_dim]) # row num = file's row num
        sample_model = RandomUnderSampler(random_state=42) # random_state = seed. undersampling: diminish majority class
        learn_X, learn_Y = sample_model.fit_sample(learn_X_reshape, learn_Y)
        learn_X = np.reshape(learn_X, [-1, seq_length, input_dim])

        matrix = []
        for v1, v2 in zip(learn_X, learn_Y):
            matrix.append([v1, v2])

        np.random.shuffle(matrix)
        learn_X = map(itemgetter(0), matrix)
        learn_Y = map(lambda x:[x], map(itemgetter(1), matrix))

        print Counter(map(lambda x:x[0], learn_Y))

        f = open('../data/seq.test.%d.csv'%(seq_length), 'r')
        test_instances = map(lambda x:x.replace('\n', '').split(','), f.readlines())
        f.close()

        np.random.shuffle(test_instances)

        test_X = []; test_Y = []

        for seq in test_instances:
            sub_x = []

            try:
                for element in seq[:-1]:
                    user_features = []
                    
                    if d_userfeatures.has_key(element):
                        user_features = d_userfeatures[element]['user'][0:2]
                    else:
                        continue

                    if user_features != []:
                        sub_x.append(np.array(user_features))

                if (len(sub_x) == seq_length):
                    test_X.append(np.array(sub_x))
                    test_Y.append(float(seq[-1]))

            except Exception as e:
                continue
        
        test_Y = map(lambda x:[x], test_Y)
        print 'size of test_Y: %d' % len(test_Y)
        print Counter(map(lambda x:x[0], test_Y))
        
        print 'Data loading Complete learn:%d, test:%d'%(len(learn_Y), len(test_Y))
        tf.reset_default_graph()


        # 2. Run RNN
        X = tf.placeholder(tf.float32, [None, seq_length, input_dim])
        Y = tf.placeholder(tf.float32, [None, 1])

        is_training = tf.placeholder(tf.bool)
        keep_prob = tf.placeholder(tf.float32)

        weights = {}
        biases = {}

        cells = []
        for _ in range(2):
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
        weights[key] = tf.Variable(tf.random_normal([hidden_size, 1]))
        biases[key] = tf.Variable(tf.random_normal([1]))
        
        optimizers = {}
        pred = []

        l1_output = tf.matmul(bn_output, weights['fc_l1']) + biases['fc_l1']
        l1_bn_output = tf.contrib.layers.batch_norm(l1_output, center=True, scale=True, is_training=is_training)

        l2_output = tf.matmul(l1_bn_output, weights['fc_l2']) + biases['fc_l2']
        l2_bn_output = tf.contrib.layers.batch_norm(l2_output, center=True, scale=True, is_training=is_training)

        logits = tf.matmul(l2_bn_output, weights['fc_l3']) + biases['fc_l3']
        labels = Y

        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        cost = tf.reduce_mean(loss)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizers = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        hypothesis = tf.sigmoid(logits)
        pred.append(tf.cast(hypothesis > 0.5, dtype=tf.float32))

        correct_pred = tf.equal(tf.round(hypothesis), Y)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for e in range(epochs):
                batch_index_start = 0
                batch_index_end = batch_size

                for i in range(int(len(learn_X)/batch_size)):
                    X_train_batch = learn_X[batch_index_start:batch_index_end]
                    Y_train_batch = learn_Y[batch_index_start:batch_index_end]

                    opt, c, o, l, acc = sess.run([optimizers, cost, outputs, logits, accuracy],
                            feed_dict={X: X_train_batch, Y: Y_train_batch, keep_prob:0.01, is_training:True})

                    batch_index_start += batch_size
                    batch_index_end += batch_size
                
                if (e % 10 == 0 and e != 0):
                    print 'epochs: %d, cost: %.8f'%(e, c)
                    # TEST
                    rst, c, h, l = sess.run([pred, cost, hypothesis, logits], feed_dict={X: test_X, Y: test_Y, keep_prob:1.0, is_training:False})

                    out = np.vstack(rst).T
                    out = out[0]

                    predicts = []
                    test_Y = map(lambda x:x[0], test_Y)

                    
                    pred_end_post = []; pred_end_comment = []
                    pred_cont_post = []; pred_cont_comment = []
                    label_end_post = []; label_end_comment = []
                    label_cont_post = []; label_cont_comment = []

                    for v1, v2 in zip(out, test_Y):
                        decision = False

                        if v1 == int(v2):
                            decision = True
                        predicts.append(decision)

                    for i, v in enumerate(out):
                        if v == 1:
                            pred_cont_post.append(test_X[i][0][0])
                            pred_cont_comment.append(test_X[i][0][1])
                        else:
                            pred_end_post.append(test_X[i][0][0])
                            pred_end_comment.append(test_X[i][0][1])

                    for i, v in enumerate(test_Y):
                        if v == 1:
                            label_cont_post.append(test_X[i][0][0])
                            label_cont_comment.append(test_X[i][0][1])
                        else:
                            label_end_post.append(test_X[i][0][0])
                            label_end_comment.append(test_X[i][0][1])

                    fpr, tpr, thresholds = roc_curve(map(int, test_Y), out)
                    print 'seq_length: %d, # predicts: %d, # corrects: %d, acc: %f, auc: %f' %(seq_length, len(predicts), len(filter(lambda x:x, predicts)), (len(filter(lambda x:x, predicts))/len(predicts)), auc(fpr,tpr))
                    print precision_recall_fscore_support(map(int, test_Y), out)
                    print 'mean_pred_cont post: %f, comment: %f'%(sum(pred_cont_post)/float(len(pred_cont_post)), sum(pred_cont_comment)/float(len(pred_cont_comment)))
                    print 'mean_pred_end post: %f, comment: %f'%(sum(pred_end_post)/float(len(pred_end_post)), sum(pred_end_comment)/float(len(pred_end_comment)))
                    print 'mean_label_cont post: %f, comment: %f'%(sum(label_cont_post)/float(len(label_cont_post)), sum(label_cont_comment)/float(len(label_cont_comment)))
                    print 'mean_label_end post: %f, comment: %f'%(sum(label_end_post)/float(len(label_end_post)), sum(label_end_comment)/float(len(label_end_comment)))
                    test_Y = map(lambda x:[x], test_Y)
            
            print 'work time: %s sec'%(time.time()-start_time)
            print '\n\n'

if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv])

