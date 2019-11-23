import tensorflow as tf
import numpy as np
import pickle
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from operator import itemgetter
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score

user_features_fields = ['posts', 'comments', 'receives']
common_features_fields = ['vader_score', 'vader', 'difficulty']
len_liwc_features = 93
len_bert_features = 768
output_dim = 1
learn_size = 60000
test_size = 20000


#def load_bertfeatures(seq_length=1):
#    return pickle.load(open('/home/jhlim/data/bertfeatures' + str(seq_length) + '.p', 'rb'))
def load_userfeatures():
    return pickle.load(open('/home/jhlim/data/userfeatures.activity.p', 'rb'))
def load_contfeatures():
    return pickle.load(open('/home/jhlim/data/contentfeatures.others.p', 'rb'))
def load_timefeatures():
    return pickle.load(open('/home/jhlim/data/temporalfeatures.p', 'rb'))

def runRNNModel(hidden_size, learning_rate, batch_size, epochs, keep_rate, seq_length=1, exclude_newbie=0, bert=0, user=0, liwc=0, cont=0, time=0, parent=0):
    test_parent = False if parent == 0 else True# for 1st -> 2nd test
    print_wrongs = False
    print ('test_parent: %r'%(test_parent))

    # Prepare the dataset
    feature_list = [bert, user, liwc, cont, time]
    length_list = [768, 3, 93, 3, 1]
#    d_bert = load_bertfeatures(seq_length)
    d_user = load_userfeatures()
    d_liwc = load_contfeatures()
    d_cont = d_liwc
    d_time = load_timefeatures()
    rnn_hidden_size = 10 
    input_dim = 0
    for i in range(len(feature_list)):
        if feature_list[i] == 1:
            input_dim += length_list[i]
    print ('seq_length: %d, input_dim: %d'%(seq_length, input_dim))

    f = open('/home/jhlim/data/seq.learn.%d.csv'%(seq_length), 'r')
    learn_instances = list(map(lambda x:x.replace('\n', '').split(','), f.readlines()))
    f.close() 
    f = open('/home/jhlim/data/seq.test.%d.csv'%(seq_length), 'r')
    test_instances = list(map(lambda x:x.replace('\n', '').split(','), f.readlines()))
    f.close()

    learn_X = []; learn_Y = []
        
    for seq in learn_instances[:learn_size]:
        sub_x = []

        try:
            for i, element in enumerate(seq[:-1]): # seq[-1] : Y. element: 't3_7dfvv'
                if False in list(map(lambda x:element in x, [d_user, d_cont, d_time])):
                    #print (list(map(lambda x:element in x, [d_user, d_cont, d_time])))
                    continue
                sub_x.append(element)
            if (len(sub_x) == seq_length):
                learn_X.append(sub_x) 
                learn_Y.append(int(seq[-1]))
        except Exception as e:
            continue

    print ('size of learn_Y: %d' % len(learn_Y))
    print (Counter(learn_Y)) # shows the number of '0' and '1'

    learn_X_reshape = np.reshape(np.array(learn_X), [-1, seq_length]) # row num = file's row num
    sample_model = RandomUnderSampler(random_state=42) # random_state = seed. undersampling: diminish majority class
    learn_X, learn_Y = sample_model.fit_sample(learn_X_reshape, learn_Y)

    test_X = []; test_Y = []
    for seq in test_instances[:test_size]:
        sub_x = []

        try:
            for i, element in enumerate(seq[:-1]):
                #if False in list(map(lambda x:element in x, [d_bert, d_user, d_cont, d_time])):
                #    continue
                if False in list(map(lambda x:element in x, [d_user, d_cont, d_time])):
                    continue
                features = []
                if feature_list[0] == 1: # Bert
                    features += d_bert[element]
                if feature_list[1] == 1: # User
                    features += d_user[element]['user']
                if feature_list[2] == 1:
                    features += d_liwc[element]['liwc']
                if feature_list[3] == 1:
                    features += d_cont[element]['cont'][0:3]
                if feature_list[4] == 1:
                    features += d_time[element]['ict']

                sub_x.append(features)
            if (len(sub_x) == seq_length):
                test_X.append(np.array(sub_x))
                test_Y.append(float(seq[-1]))

        except Exception as e:
            continue

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
            learn_X = np.array(learn_X)[:, (seq_length-2):, :].tolist()
            test_X = np.array(test_X)[:, (seq_length-2):, :].tolist()
            seq_length = seq_length-1
    if print_wrongs and seq_length > 1:
        element_list = np.array(element_list)[:, (seq_length-1):].tolist()
        print (np.array(element_list).shape)
        sentencefile = pickle.load(open('/home/jhlim/data/commentbodyfeatures.p', 'rb'))

    tf.reset_default_graph()
    tf.set_random_seed(50)
    X = tf.placeholder(tf.float32, [None, seq_length, input_dim])
    Y = tf.placeholder(tf.float32, [None, 1])

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

    #l1_output = tf.matmul(bn_output, weights['fc_l1']) + biases['fc_l1']
    l1_output = tf.nn.relu(tf.matmul(outputs, weights['fc_l1']) + biases['fc_l1']) # might move relu layer to the behind of bn
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
    #with tf.control_dependencies(update_ops): # Segmentation fault error
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
                        element = element[0]
                        features = []
                        if feature_list[0] == 1: # Bert
                            features += d_bert[element]
                        if feature_list[1] == 1: # User
                            features += d_user[element]['user']
                        if feature_list[2] == 1: # Liwc
                            features += d_liwc[element]['liwc']
                        if feature_list[3] == 1: # Cont
                            features += d_cont[element]['cont'][0:3]
                        if feature_list[4] == 1: # Time
                            features += d_time[element]['ict']
                        sub_x.append(features)

                    sequences.append(sub_x)

                X_train_batch = np.array(sequences)
                Y_train_batch = np.array(Y_train_batch)

                opt, c, o, h, l, acc = sess.run([optimizers, cost, outputs, hypothesis, logits, accuracy],
                        feed_dict={X: X_train_batch, Y: Y_train_batch, keep_prob:keep_rate, is_training:True})
                
                batch_index_start += batch_size
                batch_index_end += batch_size

            if (e % 2 == 0):
                print ('[epochs : %d, cost: %.8f]'%(e, c))

                # TEST
                rst, c, h, l = sess.run([pred, cost, hypothesis, logits], feed_dict={X: test_X, Y: test_Y, keep_prob:1.0, is_training:False})

                out = np.vstack(rst).T
                out = out[0]

                predicts = []
                test_Y = list(map(lambda x:x[0], test_Y))

                for v1, v2 in zip(out, test_Y):
                    decision = False

                    if v1 == int(v2):
                        decision = True
                    predicts.append(decision)

                print ('seq_length: %d, # predicts: %d, # corrects: %d, acc: %f, auc: %f' %(seq_length, len(predicts), len(list(filter(lambda x:x, predicts))), accuracy_score(test_Y, out), roc_auc_score(test_Y, out)))
                #print (precision_recall_fscore_support(list(map(int, test_Y)), out))
                
                test_Y = list(map(lambda x:[x], test_Y))

            if print_wrongs and e == 10:
                test_Y = list(map(lambda x:x[0], test_Y))
                print ('element size: %d, out size: %d, test_Y size: %d'%(len(element_list), len(out), len(test_Y)))

                f = open('failed_comments.txt', 'w')
                index = ['element', 'sentence', 'predicts', 'label']
                f.write('\t'.join(index) + '\n')

                for idx, tu in enumerate(zip(out, test_Y)):
                    v1, v2 = tu
                    if v1 != int(v2):
                        if element_list[idx][1] in sentencefile:
                            sentence = sentencefile[element_list[idx][1]]
                            sentence = sentence.replace('\n', ' ')
                        else:
                            sentence = 'NULL'
                        index = [element_list[idx][1], sentence, str(v1), str(v2)]
                        f.write('\t'.join(index) + '\n')

                f.close()
                test_Y = list(map(lambda x:[x], test_Y))


        print ('\n\n')



