import pickle
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from operator import itemgetter

user_features_fields = ['posts', 'comments', 'receives']
common_features_fiels = ['vader_score', 'vader', 'difficulty']
len_liwc_features = 93
len_bert_features = 768
output_dim = 1 # (range 0 to 1)
learn_size = 60000 # -1
test_size = 20000 # -1

def load_bertfeatures(seq_length=1):
    return pickle.load(open('/home/jhlim/data/bertfeatures' + str(seq_length) + '.p', 'rb'))

def load_userfeatures():
    return pickle.load(open('/home/jhlim/data/userfeatures.activity.p', 'rb'))

def load_contfeatures():
    return pickle.load(open('/home/jhlim/data/contentfeatures.others.p', 'rb'))

def load_timefeatures():
    return pickle.load(open('/home/jhlim/data/temporalfeatures.p', 'rb'))

def load_bodyfeatures():
    return pickle.load(open('/home/jhlim/data/commentbodyfeatures.p', 'rb'))

def load_w2vfeatures():
    return pickle.load(open('home/jhlim/data/contentfeatures.googlenews.nozer.p', 'rb'))

def makeLearnTestSet(seq_length=1, bert=0, user=0, liwc=0, cont=0, time=0, w2v=0,exclude_newbie=0, lr=0):
    feature_list = [bert, user, liwc, cont, time, w2v]
    length_list = [768, 3, 93, 3, 1, 300]
    d_bert = load_bertfeatures(seq_length)
    d_user = load_userfeatures()
    d_liwc = load_contfeatures()
    d_cont = d_liwc
    d_time = load_timefeatures()
    if w2v == 1:
        d_w2v = load_w2vfeatures()

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

    #np.random.shuffle(learn_instances)

    learn_X = []; learn_Y = []
        
    for seq in learn_instances[:learn_size]:
        sub_x = []

        try:
            for i, element in enumerate(seq[:-1]): # seq[-1] : Y. element: 't3_7dfvv'
                if False in list(map(lambda x:element in x, [d_bert, d_user, d_cont, d_time])):
                    continue

                features = []
                if feature_list[0] == 1:
                    features.append(d_bert[element])
                if feature_list[1] == 1:
                    features.append(d_user[element]['user'])
                if feature_list[2] == 1:
                    features.append(d_liwc[element]['liwc'])
                if feature_list[3] == 1:
                    features.append(d_cont[element]['cont'][0:3])
                if feature_list[4] == 1:
                    features.append(d_time[element]['ict'])
                if element in d_w2v:
                    features.append(d_w2v[element]['google.mean'][0])

                if features != []:
                    sub_x.append(features)

            if (len(sub_x) == seq_length):
                learn_X.append(np.array(sub_x)) # feature list
                learn_Y.append(int(seq[-1]))

        except Exception as e:
            continue

    print ('size of learn_Y: %d' % len(learn_Y))
    print (Counter(learn_Y)) # shows the number of '0' and '1'

    learn_X_reshape = np.reshape(np.array(learn_X), [-1, seq_length*input_dim]) # row num = file's row num
    sample_model = RandomUnderSampler(random_state=42) # random_state = seed. undersampling: diminish majority class
    learn_X, learn_Y = sample_model.fit_sample(learn_X_reshape, learn_Y)

    #np.random.shuffle(test_instances)

    test_X = []; test_Y = []

    for seq in test_instances[:test_size]:
        sub_x = []

        try:
            for i, element in enumerate(seq[:-1]):
                if False in list(map(lambda x:element in x, [d_bert, d_user, d_cont, d_time])):
                    continue

                features = []
                if feature_list[0] == 1:
                    features.append(d_bert[element])
                if feature_list[1] == 1:
                    features.append(d_user[element]['user'])
                if feature_list[2] == 1:
                    features.append(d_liwc[element]['liwc'])
                if feature_list[3] == 1:
                    features.append(d_cont[element]['cont'][0:3])
                if feature_list[4] == 1:
                    features.append(d_time[element]['ict'])
                if element in d_w2v:
                    features.append(d_w2v[element]['google.mean'][0])

                if features != []:
                    if exclude_newbie == 1 and d_user[element]['user'] == [0.0, 0.0, 0.0]: #w/o newbie 
                        continue
                    sub_x.append(features)


            if (len(sub_x) == seq_length):
                test_X.append(np.array(sub_x))
                test_Y.append(float(seq[-1]))

        except Exception as e:
            continue

    if lr != 1:
        learn_X = np.reshape(learn_X, [-1, seq_length, input_dim])
        matrix = []
        for v1, v2 in zip(learn_X, learn_Y):
            matrix.append([v1, v2])
        np.random.shuffle(matrix)
        learn_X = list(map(itemgetter(0), matrix))
        learn_Y = list(map(lambda x:[x], list(map(itemgetter(1), matrix))))

        test_Y = list(map(lambda x:[x], test_Y))
        print ('learn_Y: ', Counter(list(map(lambda x:x[0], learn_Y))))
        print ('test_Y: ', Counter(list(map(lambda x:x[0], test_Y))))
    else:
        test_X = np.reshape(np.array(test_X), [-1, seq_length*input_dim])
        print ('learn_Y: ', Counter(learn_Y))
        print ('test_Y: ', Counter(test_Y))

    print ('Data loading Complete learn:%d, test:%d'%(len(learn_Y), len(test_Y)))

    return learn_X, learn_Y, test_X, test_Y




