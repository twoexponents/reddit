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


def makeLearnTestSet(seq_length=1, bert=0, user=0, liwc=0, cont=0, time=0, exclude_newbie=0):
    feature_list = [bert, user, liwc, cont, time]
    length_list = [768, 3, 93, 3, 1]
    d_bert = load_bertfeatures(seq_length)
    d_user = load_userfeatures()
    d_liwc = load_contfeatures()
    d_cont = d_liwc
    d_time = load_timefeatures()
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
                if element in d_bert:
                    features.append(d_bert[element])
                else:
                    features.append([])
                if element in d_user:
                    features.append(d_user[element]['user'])
                else:
                    features.append([])
                if element in d_liwc:
                    features.append(d_liwc[element]['liwc'])
                else:
                    features.append([])
                if element in d_cont:
                    features.append(d_cont[element]['cont'][0:3])
                else:
                    features.append([])
                if element in d_time:
                    features.append(d_time[element]['ict'])
                else:
                    features.append([])

                if [] not in features:
                    sub_sub_x = np.array([])
                    for i, f in enumerate(feature_list):
                        if f == 1:
                            sub_sub_x = np.concatenate((sub_sub_x, np.array(features[i])))
                    sub_x.append(sub_sub_x)

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
    learn_Y = list(map(lambda x:[x], list(map(itemgetter(1), matrix))))

    print (Counter(list(map(lambda x:x[0], learn_Y))))

    #np.random.shuffle(test_instances)

    test_X = []; test_Y = []

    for seq in test_instances[:test_size]:
        sub_x = []

        try:
            for i, element in enumerate(seq[:-1]):
                if False in list(map(lambda x:element in x, [d_bert, d_user, d_cont, d_time])):
                    continue

                features = []
                if element in d_bert:
                    features.append(d_bert[element])
                else:
                    features.append([])
                if element in d_user:
                    features.append(d_user[element]['user'])
                else:
                    features.append([])
                if element in d_liwc:
                    features.append(d_liwc[element]['liwc'])
                else:
                    features.append([])
                if element in d_cont:
                    features.append(d_cont[element]['cont'][0:3])
                else:
                    features.append([])
                if element in d_time:
                    features.append(d_time[element]['ict'])
                else:
                    features.append([])

                if [] not in features:
                    # exclude newbie
                    if exclude_newbie == 1 and features[1] == [0.0, 0.0, 0.0]:
                        continue

                    sub_sub_x = np.array([])
                    for i, f in enumerate(feature_list):
                        if f == 1:
                            sub_sub_x = np.concatenate((sub_sub_x, np.array(features[i])))
                    sub_x.append(sub_sub_x)

            if (len(sub_x) == seq_length):
                test_X.append(np.array(sub_x))
                test_Y.append(float(seq[-1]))

        except Exception as e:
            continue

    test_Y = list(map(lambda x:[x], test_Y))
    print ('size of test_Y: %d' % len(test_Y))
    print (Counter(list(map(lambda x:x[0], test_Y))))

    print ('Data loading Complete learn:%d, test:%d'%(len(learn_Y), len(test_Y)))

    return learn_X, learn_Y, test_X, test_Y




