import tensorflow as tf
import numpy as np
from numpy.random import seed
import pickle
import random
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from operator import itemgetter
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score as ras

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

learn_size = 320000#160000
test_size = 50000#10000
term_size = 5000

def load_bertfeatures(s=3):
    return pickle.load(open('/home/jhlim/data/dynamicbertfeatures' + str(s) + '.p', 'rb'))
def load_userfeatures():
    return pickle.load(open('/home/jhlim/data/userfeatures.activity.p', 'rb'))
def load_contfeatures():
    return pickle.load(open('/home/jhlim/data/contentfeatures.others.p', 'rb'))
def load_timefeatures():
    return pickle.load(open('/home/jhlim/data/temporalfeatures.p', 'rb'))
def load_w2vfeatures():
    return pickle.load(open('/home/jhlim/data/contentfeatures.googlenews.nozero.p', 'rb'))
def load_commentbodyfeatures():
    return pickle.load(open('/home/jhlim/data/commentbodyfeatures.p', 'rb'))
def load_bert(d_bert, element):
    return d_bert[element]
def load_user(d_user, element):
    return d_user[element]['user']
def load_liwc(d_liwc, element):
    return d_liwc[element]['liwc']
def load_cont(d_cont, element):
    return d_cont[element]['cont'][0:6]
def load_time(d_time, element):
    return d_time[element]['ict']
def load_w2v(d_w2v, element):
    return d_w2v[element]['google.mean'][0].tolist()

def runRNNModel(hidden_size, learning_rate, batch_size, epochs, keep_rate, seq_length=1, exclude_newbie=0, bert=0, user=0, liwc=0, cont=0, time=0, w2v=0, seed1=10, seed2=40):
    feature_list = [bert, user, liwc, cont, time]
    length_list = [768, 3, 93, 6, 1, 300]
    test_parent = False
    print_body = True#False

    # Prepare the dataset
    d_bert = load_bertfeatures(seq_length)
    d_user = load_userfeatures()
    d_liwc = load_contfeatures()
    d_cont = d_liwc
    d_time = load_timefeatures()
    d_w2v = load_w2vfeatures() if w2v == 1 else {}
    sentencefile = load_commentbodyfeatures()

    input_dim = 0 if w2v == 0 else 300 
    input_dim += 1
    for i in range(len(feature_list)):
        if feature_list[i] == 1:
            input_dim += length_list[i]
    print ('seq_length: %d, input_dim: %d'%(seq_length, input_dim))
    rnn_hidden_size = hidden_size#input_dim

    seed(1)
    random.seed(seed1)
    f = open('/home/jhlim/data/seq.learn.less%d.csv'%(seq_length), 'r')
    lines = f.readlines()
    random.shuffle(lines)
    learn_instances = list(map(lambda x:x.replace('\n', '').split(','), lines))
    f.close() 
    f = open('/home/jhlim/data/seq.test.less%d.csv'%(seq_length), 'r')
    lines = f.readlines()
    random.shuffle(lines)
    test_instances = list(map(lambda x:x.replace('\n', '').split(','), lines))
    f.close()
    
    del lines
    learn_instances = learn_instances[:learn_size]
    test_instances = test_instances[:test_size]

    print (np.array(learn_instances).shape)
    if test_parent:
        learn_instances = list(map(lambda x: x[-2:], learn_instances))
        test_instances = list(map(lambda x: x[-2:], test_instances))
        seq_length = 1
    print (np.array(learn_instances).shape)

    print ('make element list of learn_instances.')
    learn_X = []; learn_Y = []
    for seq in learn_instances:
        sub_x = []
        flag = 0

        try:
            for i, element in enumerate(seq[:-1]): # seq[-1] : Y. element: 't3_7dfvv'
                #if False in list(map(lambda x:element in x, [d_bert, d_user, d_cont, d_time])):
                if False in list(map(lambda x:element in x, [d_bert, d_user, d_cont, d_time])):
                    flag = 1
                    break
                if w2v == 1 and element not in d_w2v:
                    flag = 1
                    break
                sub_x.append(element)

            if (flag == 0):
                temp = ['NULL'] * (seq_length - len(sub_x))
                sub_x = temp + sub_x
                learn_X.append(sub_x)
                learn_Y.append(int(seq[-1]))
        except Exception as e:
            continue

    print (np.array(learn_X).shape)

    print ('size of learn_Y: %d' % len(learn_Y))
    print (Counter(learn_Y)) # shows the number of '0' and '1'
    

    learn_X_reshape = np.reshape(np.array(learn_X), [-1, seq_length*1]) # row num = file's row num
    sample_model = RandomUnderSampler(random_state=42) # random_state = seed. undersampling: diminish majority class
    learn_X, learn_Y = sample_model.fit_sample(learn_X_reshape, learn_Y)


    test_X = []; test_Y = []
    element_list = []
    post_list = []
    
    print ('make features of test_instances.')
    for seq in test_instances:
        sub_x = []
        flag = 0
        try:
            index = 1
            for i, element in enumerate(seq[:-1]):
                if False in list(map(lambda x:element in x, [d_bert, d_user, d_cont, d_time])):
                    flag = 1
                    break
                features = []
                if feature_list[0] == 1: # Bert
                    features += load_bert(d_bert, element)
                if feature_list[1] == 1: # User
                    features += load_user(d_user, element)
                if feature_list[2] == 1:
                    features += load_liwc(d_liwc, element)
                if feature_list[3] == 1:
                    features += load_cont(d_cont, element)
                if feature_list[4] == 1:
                    features += load_time(d_time, element)
                if w2v == 1 and element in d_w2v:
                    features += load_w2v(d_w2v, element)

                if len(features) > 0:
                    features += [index]
                    index += 1
                    sub_x.append(features)

            if (flag == 0):
                temp = [[0.0] * input_dim] * (seq_length - len(sub_x))
                sub_x = temp + sub_x
                test_X.append(np.array(sub_x))
                test_Y.append(float(seq[-1]))
                element_list.append(element)
                post_list.append(seq[0])

        except Exception as e:
            continue

    learn_X = np.reshape(learn_X, [-1, seq_length, 1]) # 1: element_id (before input_dim)
    matrix = []
    for v1, v2 in zip(learn_X, learn_Y):
        matrix.append([v1, v2])
    np.random.shuffle(matrix)
    learn_X = list(map(itemgetter(0), matrix))
    learn_Y = list(map(lambda x:[x], list(map(itemgetter(1), matrix))))

    test_Y = list(map(lambda x:[x], test_Y))
    print ('learn_Y: ', Counter(list(map(lambda x:x[0], learn_Y))))
    print ('test_Y: ', Counter(list(map(lambda x:x[0], test_Y))))
    print ('Data loading Complete learn:%d, test:%d'%(len(learn_Y), len(test_Y)))

    #for item in zip(learn_X, learn_Y):
    #    print (item)

    del features, matrix, learn_instances, test_instances 

    # Start to running the model
    tf.reset_default_graph()
    tf.set_random_seed(seed2)
    X = tf.placeholder(tf.float32, [None, seq_length, input_dim])
    Y = tf.placeholder(tf.float32, [None, 1])
    #X_len = tf.placeholder(tf.int32, [batch_size])

    is_training = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32)

    weights = {}
    biases = {}

    
    cells = []
    for _ in range(1):
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_hidden_size,
                                            state_is_tuple=True,
                                            activation=tf.nn.relu)
        cells.append(cell)

    cells = tf.nn.rnn_cell.MultiRNNCell(cells)
    

    outputs, states = tf.nn.dynamic_rnn(cells, X,
            dtype=tf.float32)
    #        sequence_length=X_len)
    

    outputs = outputs[:, -1]

    bn_output = tf.contrib.layers.batch_norm(outputs, center=True, scale=True, is_training=is_training)

    key = 'fc_l1'
    #weights[key] = tf.Variable(tf.random_normal([rnn_hidden_size, hidden_size]))
    weights[key] = tf.get_variable("W1", shape=[rnn_hidden_size, hidden_size], initializer=tf.contrib.layers.xavier_initializer())
    biases[key] = tf.Variable(tf.random_normal([hidden_size]))

    key = 'fc_l2'
    #weights[key] = tf.Variable(tf.random_normal([hidden_size, hidden_size]))
    weights[key] = tf.get_variable("W2", shape=[hidden_size, hidden_size], initializer=tf.contrib.layers.xavier_initializer())
    biases[key] = tf.Variable(tf.random_normal([hidden_size]))

    key = 'fc_l3'
    #weights[key] = tf.Variable(tf.random_normal([hidden_size, 1]))
    weights[key] = tf.get_variable("W3", shape=[hidden_size, 1], initializer=tf.contrib.layers.xavier_initializer())
    biases[key] = tf.Variable(tf.random_normal([1]))
    
    optimizers = {}
    pred = []

    #10_dropout = tf.layers.dropout(outputs, rate=1-keep_prob, training=is_training)

    l1_output = tf.nn.relu(tf.matmul(bn_output, weights['fc_l1']) + biases['fc_l1'])
    #l1_output = tf.nn.relu(tf.matmul(outputs, weights['fc_l1']) + biases['fc_l1']) # might move relu layer to the behind of bn
    l1_bn_output = tf.contrib.layers.batch_norm(l1_output, center=True, scale=True, is_training=is_training)
    l1_dropout = tf.layers.dropout(l1_bn_output, rate=1-keep_prob, training=is_training)

    #l2_output = tf.matmul(l1_bn_output, weights['fc_l2']) + biases['fc_l2']
    l2_output = tf.nn.relu(tf.matmul(l1_bn_output, weights['fc_l2']) + biases['fc_l2'])
    l2_bn_output = tf.contrib.layers.batch_norm(l2_output, center=True, scale=True, is_training=is_training)
    l2_dropout = tf.layers.dropout(l2_bn_output, rate=1-keep_prob, training=is_training)

    #logits = tf.matmul(l2_bn_output, weights['fc_l3']) + biases['fc_l3']
    logits = tf.matmul(l2_dropout, weights['fc_l3']) + biases['fc_l3']
    labels = Y

    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    cost = tf.reduce_mean(loss)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    optimizers = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    optimizers = tf.group([optimizers, update_ops])

    hypothesis = tf.sigmoid(logits)
    pred.append(tf.cast(hypothesis > 0.5, dtype=tf.float32))

    correct_pred = tf.equal(tf.round(hypothesis), Y)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            # train batch by batch
            batch_index_start = 0
            batch_index_end = batch_size

            for i in range(int(len(learn_X)/batch_size)):
                X_train_batch = learn_X[batch_index_start:batch_index_end]
                Y_train_batch = learn_Y[batch_index_start:batch_index_end]
                
                sequences = []; X_len = []
                for sequence in X_train_batch:
                    sub_x = []
                    index = 1
                    
                    for i, element in enumerate(sequence):
                        element = element[0] # ex) [t3_abc] -> t3_abc
                        features = []
                        if element == 'NULL':
                            features = [0.0] * input_dim
                            #sub_x += [[0.0] * input_dim] * (seq_length - i)
                            #break
                        else:
                            if feature_list[0] == 1: # Bert
                                features += load_bert(d_bert, element)
                            if feature_list[1] == 1: # User
                                features += load_user(d_user, element)
                            if feature_list[2] == 1:
                                features += load_liwc(d_liwc, element)
                            if feature_list[3] == 1:
                                features += load_cont(d_cont, element)
                            if feature_list[4] == 1:
                                features += load_time(d_time, element)
                            features += [index]
                            index += 1

                        sub_x.append(features)

                    sequences.append(sub_x)
                    X_len.append(i)

                X_train_batch = np.array(sequences)
                Y_train_batch = np.array(Y_train_batch)
                X_length_batch = np.array(X_len)


                opt, c, o, h, l = sess.run([optimizers, cost, outputs, hypothesis, logits],
                        feed_dict={X: X_train_batch, Y: Y_train_batch, keep_prob:keep_rate, is_training:True})#, X_len: X_length_batch})
                
                batch_index_start += batch_size
                batch_index_end += batch_size

            if (e % 2 == 0 or (seq_length > 1 and bert == 1) or hidden_size > 32):
                print ('[epochs : %d, cost: %.8f]'%(e, c))

                # TEST
                rst, c, h, l = sess.run([pred, cost, hypothesis, logits], feed_dict={X: test_X, Y: test_Y, keep_prob:1.0, is_training:False})

                out = np.vstack(rst).T
                out = out[0]

                predicts = []
                test_Y = list(map(lambda x:int(x[0]), test_Y))

                for v1, v2 in zip(out, test_Y):
                    decision = False

                    if v1 == int(v2):
                        decision = True
                    predicts.append(decision)

                print ('seq_len: %d, # preds: %d, # corrs: %d, acc: %.3f, auc: %.3f, sens: %.3f, spec: %.3f' %(seq_length, len(predicts), len(list(filter(lambda x:x, predicts))), accuracy_score(test_Y, out), ras(test_Y, out), classification_report(test_Y, out, output_dict=True)['0']['recall'], classification_report(test_Y, out, output_dict=True)['1']['recall']))

                for i in range(1+len(test_Y)//term_size):
                    print ('AUC in %dth: %.3f, '%(i+1, ras(test_Y[term_size*i:term_size*(i+1)], out[term_size*i:term_size*(i+1)])), end= '')
                print()  

                test_Y = list(map(lambda x:[float(x)], test_Y))


            if print_body and e == 9:
                print ('print correct elements.')
                test_Y = list(map(lambda x:x[0], test_Y))
                f = open('result/out.txt', 'w')
                f2 = open('result/wrong.out.txt', 'w')

                for i, item in enumerate(zip(out, test_Y)):
                    v1, v2 = item
                    if v1 == int(v2):
                        f.write(element_list[i] + '\t' + str(v1) + '\t' + str(v2) + '\n')
                    else:
                        f2.write(post_list[i] + '\t' + str(v1) + '\t' + str(v2) + '\n')
                
                f.close()
                f2.close()

                test_Y = list(map(lambda x:[x], test_Y))

        print ('\n\n')



