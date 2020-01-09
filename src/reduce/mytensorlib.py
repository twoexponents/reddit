import tensorflow as tf
import numpy as np
import pickle
import random
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from operator import itemgetter
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score as ras
from numpy.random import seed

learn_size = 160000
test_size = 10000#6000

def load_bertfeatures(seq_length=1):
    return pickle.load(open('/home/jhlim/data/bertfeatures' + str(seq_length) + '.p', 'rb'))
def load_userfeatures():
    return pickle.load(open('/home/jhlim/data/userfeatures.activity.p', 'rb'))
def load_contfeatures():
    return pickle.load(open('/home/jhlim/data/contentfeatures.others.p', 'rb'))
def load_timefeatures():
    return pickle.load(open('/home/jhlim/data/temporalfeatures.p', 'rb'))
def load_w2vfeatures():
    return pickle.load(open('/home/jhlim/data/contentfeatures.googlenews.nozero.p', 'rb'))
def load_glovefeatures():
    return pickle.load(open('/home/jhlim/data/contentfeatures.w2v.glove.p', 'rb'))
def load_commentbodyfeatures():
    return pickle.load(open('/home/jhlim/data/commentbodyfeatures.p', 'rb'))
def load_bert(d_bert, element):
    return d_bert[element]
def load_user(d_user, element):
    return d_user[element]['user'][0:3]
def load_liwc(d_liwc, element):
    return d_liwc[element]['liwc']
def load_cont(d_cont, element):
    return d_cont[element]['cont'][5:6]
def load_time(d_time, element):
    return d_time[element]['ict']
def load_w2v(d_w2v, element):
    return d_w2v[element]['google.mean'][0].tolist()
def load_glove(d_glove, element):
    return d_glove[element]['glove.mean'][0].tolist()

def runRNNModel(hidden_size, learning_rate, batch_size, epochs, keep_rate, seq_length=1, exclude_newbie=0, bert=0, user=0, liwc=0, cont=0, time=0, w2v=0, glove=0):
    feature_list = [bert, user, liwc, cont, time]
    length_list = [768, 3, 93, 1, 1, 300, 100]
    test_parent = True # for 1st -> 2nd test
    remove_long = False#True
    print_body = True
    print ('test_parent: %r'%(test_parent))

    # Prepare the dataset
    d_bert = load_bertfeatures(seq_length)
    d_user = load_userfeatures()
    d_liwc = load_contfeatures()
    d_cont = d_liwc
    d_time = load_timefeatures()
    d_w2v = load_w2vfeatures() #if w2v == 1 else {}
    d_glove = load_glovefeatures() if glove == 1 else {}
    sentencefile = load_commentbodyfeatures()

    input_dim = 0
    if w2v == 1:
        input_dim = 300
    elif glove == 1:
        input_dim = 100
    for i in range(len(feature_list)):
        if feature_list[i] == 1:
            input_dim += length_list[i]
    print ('seq_length: %d, input_dim: %d'%(seq_length, input_dim))
    #rnn_hidden_size = input_dim#hidden_size
    rnn_hidden_size = hidden_size 

    f = open('/home/jhlim/data/seq.learn.%d.csv'%(seq_length), 'r')
    learn_instances = list(map(lambda x:x.replace('\n', '').split(','), f.readlines()))
    f.close() 
    f = open('/home/jhlim/data/seq.test.%d.csv'%(seq_length), 'r')
    test_instances = list(map(lambda x:x.replace('\n', '').split(','), f.readlines()))
    f.close()

    learn_X = []; learn_Y = []
    fal = 0
    for seq in learn_instances[:learn_size]:
        sub_x = []

        try:
            for i, element in enumerate(seq[:-1]): # seq[-1] : Y. element: 't3_7dfvv'
                if False in list(map(lambda x:element in x, [d_bert, d_user, d_cont, d_time])):
                    fal += 1
                    continue
                if remove_long:
                    if element in sentencefile and len(sentencefile[element]) > 100:
                        continue
                if w2v == 1 and element not in d_w2v:
                    continue
                if glove == 1 and element not in d_glove:
                    continue
                sub_x.append(element)
            if (len(sub_x) == seq_length):
                learn_X.append(sub_x) 
                learn_Y.append(int(seq[-1]))
        except Exception as e:
            continue
    #print ('fal: ', fal)

    print ('size of learn_Y: %d' % len(learn_Y))
    print (Counter(learn_Y)) # shows the number of '0' and '1'

    learn_X_reshape = np.reshape(np.array(learn_X), [-1, seq_length]) # row num = file's row num
    sample_model = RandomUnderSampler(random_state=42) # random_state = seed. undersampling: diminish majority class
    learn_X, learn_Y = sample_model.fit_sample(learn_X_reshape, learn_Y)

    test_X = []; test_Y = []
    element_list = []
    post_list = []    
    for seq in test_instances[:test_size]:
        sub_x = []
        try:
            for i, element in enumerate(seq[:-1]):
                if False in list(map(lambda x:element in x, [d_bert, d_user, d_cont, d_time, d_w2v])):
                    continue
                if remove_long:
                    if element in sentencefile and len(sentencefile[element]) > 100:
                        continue
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
                if glove == 1 and element in d_glove:
                    features += load_glove(d_glove, element)

                if features != []:
                    sub_x.append(features)

            if (len(sub_x) == seq_length):
                test_X.append(np.array(sub_x))
                test_Y.append(float(seq[-1]))
                element_list.append(element)
                post_list.append(seq[0])

        except Exception as e:
            continue

    #seed(1)
    #random.seed(30)
    seed(10)
    random.seed(40)
    learn_X = np.reshape(learn_X, [-1, seq_length, 1])
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



    # Start to running the model

    if test_parent:
        if seq_length != 3:
            learn_X = np.array(learn_X)[:, (seq_length-1):, :].tolist()
            test_X = np.array(test_X)[:, (seq_length-1):, :].tolist()
            seq_length = 1
        else:
            learn_X = np.array(learn_X)[:, 1:, :].tolist()
            test_X = np.array(test_X)[:, 1:, :].tolist()
            seq_length = seq_length-1

    tf.reset_default_graph()
    tf.set_random_seed(40)
    X = tf.placeholder(tf.float32, [None, seq_length, input_dim])
    Y = tf.placeholder(tf.float32, [None, 1])

    is_training = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32)

    weights = {}
    biases = {}

    
    cells = []
    for _ in range(1):
        #cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units=rnn_hidden_size)
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_hidden_size,
                                            state_is_tuple=True,
                                            activation=tf.nn.relu
                                            )
        #cell = tf.contrib.rnn.BasicRNNCell(num_units=rnn_hidden_size,
        #                                    activation=tf.nn.relu
        #                                    )
        cells.append(cell)

    cells = tf.nn.rnn_cell.MultiRNNCell(cells)
    

    outputs, states = tf.nn.dynamic_rnn(cells, X,
            dtype=tf.float32)
    

    outputs = outputs[:, -1]

    bn_output = tf.contrib.layers.batch_norm(outputs, center=True, scale=True, is_training=is_training)

    key = 'fc_l1'
    weights[key] = tf.Variable(tf.random_normal([rnn_hidden_size, hidden_size]))
    biases[key] = tf.Variable(tf.random_normal([hidden_size]))

    key = 'fc_l2'
    weights[key] = tf.Variable(tf.random_normal([hidden_size, hidden_size]))
    biases[key] = tf.Variable(tf.random_normal([hidden_size]))

    key = 'fc_l3'
    weights[key] = tf.Variable(tf.random_normal([hidden_size, 1]))
    biases[key] = tf.Variable(tf.random_normal([1]))
    
    optimizers = {}
    pred = []

    #10_dropout = tf.layers.dropout(outputs, rate=1-keep_prob, training=is_training)

    l1_output = tf.nn.relu(tf.matmul(bn_output, weights['fc_l1']) + biases['fc_l1'])
    #l1_output = tf.nn.relu(tf.matmul(outputs, weights['fc_l1']) + biases['fc_l1'])
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
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            # train batch by batch
            batch_index_start = 0
            batch_index_end = batch_size

            for i in range(int(len(learn_X)/batch_size)):
                X_train_batch = learn_X[batch_index_start:batch_index_end]
                Y_train_batch = learn_Y[batch_index_start:batch_index_end]

                sequences = []
                for sequence in X_train_batch:
                    sub_x = []
                    for element in sequence:
                        element = element[0] # ex) [t3_abc] -> t3_abc
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
                        if glove == 1 and element in d_glove:
                            features += load_glove(d_glove, element)
                        
                        sub_x.append(features)

                    sequences.append(sub_x)

                X_train_batch = np.array(sequences)
                Y_train_batch = np.array(Y_train_batch)


                opt, c, o, h = sess.run([optimizers, cost, outputs, hypothesis],
                        feed_dict={X: X_train_batch, Y: Y_train_batch, keep_prob:keep_rate, is_training:True})
                
                batch_index_start += batch_size
                batch_index_end += batch_size

            #if (e % 2 == 0 or (seq_length > 1 and bert == 1) or hidden_size > 32):
            if (e % 2 == 0):
                print ('[epochs : %d, cost: %.8f]'%(e, c))

                # TEST
                rst, c = sess.run([pred, cost], feed_dict={X: test_X, Y: test_Y, keep_prob:1.0, is_training:False})

                out = np.vstack(rst).T
                out = out[0]

                predicts = []
                test_Y = list(map(lambda x:int(x[0]), test_Y))

                for v1, v2 in zip(out, test_Y):
                    decision = False

                    if v1 == int(v2):
                        decision = True
                    predicts.append(decision)

                print ('seq_length: %d, # predicts: %d, # corrects: %d, acc: %.3f, auc: %.3f, sens: %.3f, spec: %.3f' %(seq_length, len(predicts), len(list(filter(lambda x:x, predicts))), accuracy_score(test_Y, out), ras(test_Y, out), classification_report(test_Y, out, output_dict=True)['0']['recall'], classification_report(test_Y, out, output_dict=True)['1']['recall']))
                #print (classification_report(test_Y, out, labels=[0, 1]))

                for i in range(1+len(test_Y)//5000):
                    print ('AUC in %dth: %.3f, '%(i+1, ras(test_Y[5000*i:5000*(i+1)], out[5000*i:5000*(i+1)])), end= '')
                print ()

                test_Y = list(map(lambda x:[x], test_Y))

            if print_body and e == 22:
                print ('print correct elements.')
                test_Y = list(map(lambda x:x[0], test_Y))
                f = open('result/out.txt', 'w')
                f2 = open('result/testset.txt', 'w')
                f_post = open('result/wrong.out.txt', 'w')

                for i, item in enumerate(zip(out, test_Y)):
                    v1, v2 = item
                    f2.write(element_list[i] + '\t' + str(v2) + '\n')

                    if v1 == int(v2):
                        f.write(str(i) + '\t' + element_list[i] + '\t' + str(v1) + '\t' + str(v2) + '\n')
                    else:
                        f_post.write(post_list[i] + '\t' + str(v1) + '\t' + str(v2) + '\n')
                
                f.close()
                f2.close()

                test_Y = list(map(lambda x:[x], test_Y))

        print ('\n\n')



