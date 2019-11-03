import tensorflow as tf
import numpy as np
import sys
import pickle

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score


def runRNNModel(seq_length, learn_X, learn_Y, test_X, test_Y, input_dim, output_dim, hidden_size, learning_rate, batch_size, epochs, keep_rate, element_list = None):
    test_parent = True # for 1st -> 2nd test
    print_wrongs = False

    if test_parent:
        if seq_length != 3:
            learn_X = np.array(learn_X)[:, (seq_length-1):, :].tolist()
            test_X = np.array(test_X)[:, (seq_length-1):, :].tolist()
            seq_length = 1
        else:
            learn_X = np.array(learn_X)[:, (seq_length-2):, :].tolist()
            test_X = np.array(test_X)[:, (seq_length-2):, :].tolist()
            seq_length = seq_length-1
    if print_wrongs:
        element_list = np.array(element_list)[:, 1:].tolist()
        print (np.array(element_list).shape)

    sentencefile = pickle.load(open('/home/jhlim/data/commentbodyfeatures.p', 'rb'))


    tf.reset_default_graph()
    tf.set_random_seed(50)
    X = tf.placeholder(tf.float32, [None, seq_length, input_dim])
    Y = tf.placeholder(tf.int32, [None])

    is_training = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32)

    weights = {}
    biases = {}

    cells = []
    for _ in range(1):
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size,
                                            state_is_tuple=True,
                                            activation=tf.nn.relu)
        cells.append(cell)

    cells = tf.nn.rnn_cell.MultiRNNCell(cells)

    outputs, states = tf.nn.dynamic_rnn(cells, X,
            dtype=tf.float32)

    outputs = outputs[:, -1]

    bn_output = tf.contrib.layers.batch_norm(outputs, center=True, scale=True, is_training=is_training)

    key = 'fc_l1'
    weights[key] = tf.Variable(tf.random_normal([hidden_size, hidden_size]))
    biases[key] = tf.Variable(tf.random_normal([hidden_size]))

    key = 'fc_l2'
    weights[key] = tf.Variable(tf.random_normal([hidden_size, hidden_size]))
    biases[key] = tf.Variable(tf.random_normal([hidden_size]))

    key = 'fc_l3'
    weights[key] = tf.Variable(tf.random_normal([hidden_size, 3]))
    biases[key] = tf.Variable(tf.random_normal([3]))
    
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
    labels = tf.one_hot(Y, depth=output_dim) 

    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
    cost = tf.reduce_mean(loss)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #with tf.control_dependencies(update_ops): # Segmentation fault error
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
                        feed_dict={X: X_train_batch, Y: Y_train_batch, keep_prob:keep_rate, is_training:True})
                
                batch_index_start += batch_size
                batch_index_end += batch_size

            if (e % 2 == 0):
                print ('epochs : %d, cost: %.8f'%(e, c))
                # TEST
                rst, c = sess.run([pred, cost], feed_dict={X: test_X, Y: test_Y, keep_prob:1.0, is_training:False})

                out = np.vstack(rst).T

                predicts = []

                for v1, v2 in zip(out, test_Y):
                    decision = False

                    if v1 == int(v2):
                        decision = True
                    predicts.append(decision)

                print ('seq_length: %d, # predicts: %d, # corrects: %d, acc: %f' %(seq_length, len(predicts), len(list(filter(lambda x:x, predicts))), accuracy_score(test_Y, out)))
                #print (precision_recall_fscore_support(list(map(int, test_Y)), out))
                

            if print_wrongs and e == 10:
                print ('element size: %d, out size: %d, test_Y size: %d'%(len(element_list), len(out), len(test_Y)))

                f = open('failed_comments.txt', 'w')
                index = ['element', 'sentence', 'predicts', 'label']
                f.write('\t'.join(index) + '\n')

                for idx, tu in enumerate(zip(out, test_Y)):
                    v1, v2 = tu
                    if v1 != int(v2):
                        if element_list[idx][0] in sentencefile:
                            sentence = sentencefile[element_list[idx][0]]
                            sentence = sentence.replace('\n', ' ')
                        else:
                            sentence = 'NULL'
                        index = [element_list[idx][0], sentence, str(v1), str(v2)]
                        f.write('\t'.join(index) + '\n')

                f.close()


        print ('\n\n')



