import tensorflow as tf
import numpy as np
import pickle
import random
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from operator import itemgetter
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score

learn_size = 140000
test_size = 20000

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
    return d_cont[element]['cont'][2:3]
def load_time(d_time, element):
    return d_time[element]['ict']
def load_w2v(d_w2v, element):
    return d_w2v[element]['google.mean'][0].tolist()

def runLRModel(seq_length=1, exclude_newbie=0, bert=0, user=0, liwc=0, cont=0, time=0, w2v=0):
    feature_list = [bert, user, liwc, cont, time]
    length_list = [768, 3, 93, 1, 1, 300]
    test_parent = False#True
    print_body = False

    # Prepare the dataset
    d_bert = load_bertfeatures(seq_length)
    d_user = load_userfeatures()
    d_liwc = load_contfeatures()
    d_cont = d_liwc
    d_time = load_timefeatures()
    d_w2v = load_w2vfeatures() if w2v == 1 else {}
    sentencefile = load_commentbodyfeatures()

    input_dim = 0 if w2v == 0 else 300 
    for i in range(len(feature_list)):
        if feature_list[i] == 1:
            input_dim += length_list[i]
    print ('seq_length: %d, input_dim: %d'%(seq_length, input_dim))

    f = open('/home/jhlim/data/seq.learn.less%d.csv'%(seq_length), 'r')
    learn_instances = list(map(lambda x:x.replace('\n', '').split(','), f.readlines()))
    f.close() 
    f = open('/home/jhlim/data/seq.test.less%d.csv'%(seq_length), 'r')
    test_instances = list(map(lambda x:x.replace('\n', '').split(','), f.readlines()))
    f.close()

    random.seed(300)
    random.shuffle(learn_instances)
    learn_instances = learn_instances[:200000]
    random.shuffle(test_instances)
    test_instances = test_instances[:50000]


    if test_parent:
        learn_instances = list(map(lambda x: x[-2:], learn_instances))
        test_instances = list(map(lambda x: x[-2:], test_instances))

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
                features = []
                if feature_list[0] == 1:
                    features += d_bert[element]
                if feature_list[1] == 1:
                    features += d_user[element]['user']
                if feature_list[2] == 1:
                    features += d_liwc[element]['liwc']
                if feature_list[3] == 1:
                    features += d_cont[element]['cont'][2:3]
                if feature_list[4] == 1:
                    features += d_time[element]['ict']
                if element in d_w2v:
                    features += d_w2v[element]['google.mean'][0]
                if features != []:
                    sub_x.append(features)

            if (flag == 0):
                temp = [[0.0] * input_dim] * (seq_length - len(sub_x))
                sub_x += temp
                learn_X.append(sub_x)
                learn_Y.append(int(seq[-1]))
        except Exception as e:
            continue

    print ('size of learn_Y: %d' % len(learn_Y))
    print (Counter(learn_Y)) # shows the number of '0' and '1'
    

    learn_X_reshape = np.reshape(np.array(learn_X), [-1, seq_length*input_dim]) # row num = file's row num
    sample_model = RandomUnderSampler(random_state=42) # random_state = seed. undersampling: diminish majority class
    learn_X, learn_Y = sample_model.fit_sample(learn_X_reshape, learn_Y)


    test_X = []; test_Y = []
    element_list = []
    
    for seq in test_instances[:test_size]:
        sub_x = []
        flag = 0
        try:
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
                    if exclude_newbie == 1 and d_user[element]['user'] == [0.0, 0.0, 0.0]:
                        continue
                    sub_x.append(features)

            if (flag == 0):
                temp = [[0.0] * input_dim] * (seq_length - len(sub_x))
                sub_x += temp
                test_X.append(np.array(sub_x))
                test_Y.append(float(seq[-1]))
                element_list.append(element)

        except Exception as e:
            continue

    test_X = np.reshape(np.array(test_X), [-1, seq_length*input_dim])
    print ('learn_Y: ', Counter(learn_Y))
    print ('test_Y: ', Counter(test_Y))

    print ('Data loading Complete learn:%d, test:%d'%(len(learn_Y), len(test_Y)))

    del features, learn_instances, test_instances 

    input_dim = int(len(learn_X[0]) / seq_length)

    # To test 1st comment -> 2nd comment
    if test_parent:
        learn_X = np.array(learn_X)[:, (seq_length-1)*input_dim:].tolist()
        test_X = np.array(test_X)[:, (seq_length-1)*input_dim:].tolist()
        print (np.array(learn_X).shape)

    clf = LogisticRegression(penalty='l2')
    clf.fit(learn_X, learn_Y)

    out = list(map(int, clf.predict(test_X).tolist()))
    predicts = []

    test_Y = [int(i) for i in test_Y]

    for v1, v2 in zip(out, test_Y):
      decision = False

      if v1 == int(v2):
        decision = True
      predicts.append(decision)

    y_true = list(map(int, test_Y))

    n = len(predicts)
    corrects = len(list(filter(lambda x:x, predicts)))

    acc = float(corrects)/n
    prec, rec, f1, support, = precision_recall_fscore_support(y_true, out)
    fpr, tpr, thresholds = roc_curve(y_true, out)
    auc_v = auc(fpr, tpr)
    ap = average_precision_score (y_true, out)
    #print ('%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f'%(
    #  seq_length, n, corrects, acc,
    #  prec[0], prec[1], rec[0], rec[1], f1[0], f1[1], auc_v, ap))
    print ('seq_length: %d, n: %d, corrects: %d, acc: %.3f, auc_v: %.3f, prfs: '%(
        seq_length, n, corrects, accuracy_score(test_Y, out), roc_auc_score(test_Y, out)), precision_recall_fscore_support(test_Y, out))

