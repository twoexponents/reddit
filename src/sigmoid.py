import tensorflow as tf
import numpy as np
import sys
import cPickle as pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from collections import Counter
from operator import itemgetter

common_features_fields = ['vader_score', 'vader', 'difficulty']
post_features_fields = ['pub_1h', 'pub_hd', 'pub_1d', 'max_similarity_1h',
	'max_similarity_hd', 'max_similarity_1d', 'pub_time_0', 'pub_time_1',
	'pub_time_2', 'pub_time_3', 'pub_time_4', 'pub_time_5', 'pub_time_6',
	'pub_time_7', 'pub_time_8', 'pub_time_9', 'pub_time_10', 'pub_time_11',
        'pub_time_12', 'pub_time_13', 'pub_time_14', 'pub_time_15', 'pub_time_16',
        'pub_time_17', 'pub_time_18', 'pub_time_19', 'pub_time_20', 'pub_time_21',
        'pub_time_22', 'pub_time_23']
comment_features_fields = ['similarity_post', 'similarity_parent', 'inter_comment_time', 'prev_comments']
user_features_fields = ['posts', 'comments', 'convs', 'entropy_conv']

cont_features_fields = common_features_fields + post_features_fields

len_liwc_features = 93
len_w2v_features = 300

#input_dim = len(cont_features_fields) + len(user_features_fields) + len_liwc_features + len_w2v_features
input_dim = len(user_features_fields)

output_dim = 1 # (range 0 to 1)
hidden_size = 100
learning_rate = 0.01
batch_size = 100
epochs = 5

# jhlim: How about to use data normalization by MinMaxScaler?

def main(argv):
    # 1.1 load feature dataset
    #d_features = pickle.load(open('../data/contentfeatures.others.p', 'r'))
    #d_w2vfeatures = pickle.load(open('../data/contentfeatures.googlenews.posts.p', 'r'))
    d_userfeatures = pickle.load(open('../data/userfeatures.activity.p', 'r'))

    print 'features are loaded'

    for seq_length in xrange(5, 6):
        f = open('../data/seq.learn.%d.csv'%(seq_length), 'r')
        learn_instances = map(lambda x:x.replace('\n', '').split(','), f.readlines())
        f.close()

        np.random.shuffle(learn_instances)

        learn_X = []; learn_Y = []
        cnt = 0;
        for seq in learn_instances:
            sub_x = []

            #if seq[-1] == '0':
            #   continue

            try:
                for element in seq[:-1]: # seq[-1] : Y. element: 't3_7dfvv'
                    #cont_features = [0.0]*len(cont_features_fields)
                    #liwc_features = [0.0]*len_liwc_features
                    #w2v_features = [0.0]*len_w2v_features
                    user_features = [0.0]*len(user_features_fields)

                    # if the features of element ID exist, bring it.
                    #if (cnt%1000 == 0):
                    #    print cnt, " ", element
                    #cnt+=1;
                    '''
                    if d_features.has_key(element):
                        cont_features = d_features[element]['cont']
                        liwc_features = d_features[element]['liwc']
                        w2v_features = d_w2vfeatures[element]['glove.mean'][0]
                        user_features = d_userfeatures[element]['user']
                    else:
                        print 'It does not have the element.'
                    '''
                    if d_userfeatures.has_key(element):
                        user_features = d_userfeatures[element]['user']
                    else:
                        continue
                    # append each element's features of seq to the sub_x
                    #sub_x.append(np.array(cont_features+liwc_features+w2v_features.tolist()+user_features))
                    sub_x.append(np.array(user_features))
                    #sub_x.append(np.array(cont_features+liwc_features+user_features))

                if user_features[0] != 0:
                    learn_X.append(np.array(sub_x)) # feature list
                    learn_Y.append(float(seq[-1]))

            except Exception, e:
                # print e
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

        #print "learn_X: ", learn_X
        #print "learn_Y: ", learn_Y

        print Counter(map(lambda x:x[0], learn_Y))

        f = open('../seq.learn.%d.csv'%(seq_length), 'r')
        test_instances = map(lambda x:x.replace('\n', '').split(','), f.readlines())
        f.close()

        np.random.shuffle(test_instances)

        test_X = []; test_Y = []

        for seq in test_instances:
            sub_x = []

            try:
                for element in seq[:-1]:
                    #cont_features = [0.0]*len(cont_features_fields)
                    #liwc_features = [0.0]*len_liwc_features
                    #w2v_features = [0.0]*len_w2v_features
                    user_features = [0.0]*len(user_features_fields)
                    '''
                    if d_features.has_key(element):
                        cont_features = d_features[element]['cont']
                        liwc_features = d_features[element]['liwc']
                        w2v_features = d_w2vfeatures[element]['glove.tfidf'][0]
                        user_features = d_userfeatures[element]['user']
                    else:
                        print 'It does not have the element.'
                    '''
                    if d_userfeatures.has_key(element):
                        user_features = d_userfeatures[element]['user']
                    else:
                        continue

                    #sub_x.append(np.array(cont_features+liwc_features+w2v_features.tolist()+user_features))
                    sub_x.append(np.array(user_features))

                if user_features[0] != 0:
                    test_X.append(np.array(sub_x))
                    test_Y.append([float(seq[-1])])

            except Exception, e:
                continue

        print 'Data loading Complete learn:%d, test:%d'%(len(learn_Y), len(test_Y))
        tf.reset_default_graph()

        #print "test_Y: ", test_Y

        # 2. Run RNN
        X = tf.placeholder(tf.float32, [None, seq_length, input_dim])
        Y = tf.placeholder(tf.float32, [None, 1])

        is_training = tf.placeholder(tf.bool)

        # sequence_length = seq_length
        keep_prob = tf.placeholder(tf.float32)

        weights = {}
        biases = {}

        cells = []
        for _ in range(2):
            #cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size,
            #                                       state_is_tuple=True,
            #                                       activation=tf.nn.relu)
            cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=hidden_size,
                                                        activation=tf.nn.relu,
                                                        dropout_keep_prob=keep_prob) # Layer Normalization. num_units: ouput size

            cells.append(cell)

        cells = tf.nn.rnn_cell.MultiRNNCell(cells) # stackedRNN

        outputs, states = tf.nn.dynamic_rnn(cells, X,
                dtype=tf.float32) # called RNN driver

        outputs = outputs[:, -1]

        bn_output = tf.contrib.layers.batch_norm(outputs, center=True, scale=True, is_training=is_training)

        # three-level MLP
        key = 'fc_l1'
        weights[key] = tf.Variable(tf.random_normal([hidden_size, 100]))
        biases[key] = tf.Variable(tf.random_normal([100]))

        key = 'fc_l2'
        weights[key] = tf.Variable(tf.random_normal([100, 100]))
        biases[key] = tf.Variable(tf.random_normal([100]))

        key = 'fc_l3'
        weights[key] = tf.Variable(tf.random_normal([100, 1]))
        biases[key] = tf.Variable(tf.random_normal([1]))
        
        optimizers = {}
        pred = []

        # Using dropout -> Fail
        # Using only batch normalization (w/ Relu) -> Fail
        # Using only batch normalization (w/o relu) -> better

        #10_dropout = tf.layers.dropout(outputs, rate=1-keep_prob, training=is_training)

        l1_output = tf.matmul(bn_output, weights['fc_l1']) + biases['fc_l1']
        #l1_output = tf.nn.relu(tf.matmul(outputs, weights['fc_l1']) + biases['fc_l1']) # might move relu layer to the behind of bn
        l1_bn_output = tf.contrib.layers.batch_norm(l1_output, center=True, scale=True, is_training=is_training)
        #l1_dropout = tf.layers.dropout(l1_output, rate=1-keep_prob, training=is_training)

        l2_output = tf.matmul(l1_bn_output, weights['fc_l2']) + biases['fc_l2']
        #l2_output = tf.nn.relu(tf.matmul(l1_bn_output, weights['fc_l2']) + biases['fc_l2']
        l2_bn_output = tf.contrib.layers.batch_norm(l2_output, center=True, scale=True, is_training=is_training)

        #logits = tf.matmul(l2_bn_output, weights['fc_l3']) + biases['fc_l3']
        #labels = tf.one_hot(Y, depth=output_dim) # output_dim: 2
        
        logits = tf.matmul(l2_bn_output, weights['fc_l3']) + biases['fc_l3']
        labels = Y

        logits = tf.contrib.layers.batch_norm(logits, center=True, scale=True, is_training=is_training)
        #labels = tf.one_hot(Y, depth=output_dim)

        #loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
        #cost = tf.reduce_mean(loss)
        #optimizers = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        hypothesis = tf.sigmoid(logits)
        cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
        optimizers = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        #optimizers = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

        pred.append(tf.cast(hypothesis > 0.5, dtype=tf.float32))
        #pred.append(tf.argmax(tf.nn.softmax(logits), 1))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            count = 0
            for _ in range(epochs):

                # train batch by batch
                batch_index_start = 0
                batch_index_end = batch_size

                for i in range(int(len(learn_X)/batch_size)):
                    X_train_batch = learn_X[batch_index_start:batch_index_end]
                    Y_train_batch = learn_Y[batch_index_start:batch_index_end]

                    opt, c, o, h, l = sess.run([optimizers, cost, outputs, hypothesis, outputs],
                            feed_dict={X: X_train_batch, Y: Y_train_batch, keep_prob:0.01, is_training:True})
                    
                    print 'iteration : %d, cost: %.8f'%(count, c)
                    print 'hypothesis : ', h
                    print 'logits : ', l
                    #print 'outputs: ', outputs

                    batch_index_start += batch_size
                    batch_index_end += batch_size
                    count += 1

            #print "train_Y: ", Y_train_batch
            #print "test_Y: ", test_Y

            rst = sess.run(pred, feed_dict={X: test_X, Y:test_Y, keep_prob:1.0, is_training:False})

            out = np.vstack(rst).T

            out = out[0]

            #temp = map(lambda x:x[0], out.tolist())
            print '# predict', Counter(out)
            print '# test', Counter(map(lambda x:x[0], test_Y))

            predicts = []
            test_Y = map(lambda x:x[0], test_Y)

            f = open('../result/result.rnn.%d.tsv'%(seq_length), 'w')
            for v1, v2 in zip(out, test_Y):
                f.write('%d,%s\n'%(v1, v2))
                decision = False

                if v1 == int(v2):
                    decision = True
                predicts.append(decision)

            print 'seq_length: %d, # predicts: %d, # corrects: %d' %(seq_length, len(predicts), len(filter(lambda x:x, predicts)))
            print precision_recall_fscore_support(map(int, test_Y), out)
            print '\n\n'

            f.close()


if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv])

