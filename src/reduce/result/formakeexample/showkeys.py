import pickle
from operator import itemgetter
from statistics import mean
import string
import re
from math import pow, sqrt

exclude = set(string.punctuation)

def load_commentbodyfeatures():
    return pickle.load(open('/home/jhlim/data/commentbodyfeatures.p', 'rb'))

def load_keydict():
    sentencefile = load_commentbodyfeatures()

    f1 = open('testset.txt', 'r')
    f2 = open('w2v.out.txt', 'r')
    f3 = open('bert.out.txt', 'r')

    # 1) make set of rf that contains elements' indexes predicting correctly
    idx_rf = set()
    rf_lines = f2.readlines()
    print (len(rf_lines))
    for item in rf_lines:
        elements = item.replace('\n', '').split()
        idx_rf.add(int(elements[0]))

    # 2) make set of rnn that contains elements' indexes predicting correctly
    idx_rnn = set()
    rnn_lines = f3.readlines()
    for item in rnn_lines:
        elements = item.replace('\n', '').split()
        idx_rnn.add(int(elements[0]))

    # 3) make word dicts of three files
    d_test = {}; d_rf = {}; d_rnn = {};
    d_keys = {}
    test_lines = f1.readlines()
    for idx, item in enumerate(test_lines):
        elements = item.replace('\n', '').split('\t')
        id = elements[0]; label = int(elements[1])
        pred_rf = label; pred_rnn = label;

        if id not in sentencefile:
            continue
        if idx not in idx_rf:
            pred_rf = (pred_rf + 1) % 2
        if idx not in idx_rnn:
            pred_rnn = (pred_rnn + 1) % 2
        #print (label, pred_rf, pred_rnn)

        sentence = ''
        for c in sentencefile[id]:
            if c not in exclude:
                sentence += c

        words = set()
        for word in sentence.split(' '):
            words.add(word)

        for word in words:
            if word in d_test:
                d_keys[word].append((id, label))
                d_test[word].append(label)
                d_rf[word].append(pred_rf)
                d_rnn[word].append(pred_rnn)
            else:
                d_keys[word] = [(id, label)]
                d_test[word] = [label]
                d_rf[word] = [pred_rf]
                d_rnn[word] = [pred_rnn]

    return d_keys, d_test, d_rf, d_rnn
    '''
    print ('dict len: ', len(d_test.items()))


    diff_lst = []
    for item in d_test.items():
        key, value = item
        #print (key, value)
        if len(value) < 10:
            continue
        m_test = mean(value)
        m_rf = mean(d_rf[key])
        m_rnn = mean(d_rnn[key])
        diff_1 = sqrt(pow(m_test - m_rf, 2))
        diff_2 = sqrt(pow(m_test - m_rnn, 2))

        f_test = 0 if m_test < 0.5 else 1
        f_rf = 0 if m_rf < 0.5 else 1
        f_rnn = 0 if m_rnn < 0.5 else 1
        
        diff_lst.append((key, len(value), sqrt(pow(diff_1 - diff_2, 2)), f_test, f_rf, f_rnn, m_test, m_rf, m_rnn))

    sorted_diff_lst = sorted(diff_lst, key=itemgetter(2))
    sorted_diff_lst.reverse()

    print ('word, cnt, diff, m_test, m_rf, m_rnn')
    for element in sorted_diff_lst[:100]:
        if element[3] == element[5] and element[4] != element[5]:
            print (element)
    '''


    f1.close()
    f2.close()
    f3.close()
