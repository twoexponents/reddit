import tensorflow as tf
import numpy as np
import sys
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 

from collections import Counter
from myrflib import runRFModel

user_features_fields = ['posts', 'comments']
len_liwc_features = 93
len_bert_features = 768

input_dim = len_liwc_features + len_bert_features

output_dim = 2 # (0, 1)


def main(argv):
  exclude_newbie = 0; input_length = 1
  if len(sys.argv) >= 3:
    exclude_newbie = int(sys.argv[2])
  if len(sys.argv) >= 2:
    input_length = int(sys.argv[1])
  print ('sequence len: %d' % (input_length))
  print ('exclude_newbie: %d'%(exclude_newbie))

  # 1.1 load feature dataset
  d_features = pickle.load(open('/home/jhlim/data/contentfeatures.others.p', 'rb'))
  d_userfeatures = pickle.load(open('/home/jhlim/data/userfeatures.activity.p', 'rb'))
  d_bertfeatures1 = pickle.load(open('/home/jhlim/data/bertfeatures.p', 'rb'))
  d_bertfeatures2 = pickle.load(open('/home/jhlim/data/bertfeatures2.p', 'rb'))

  print ('features are loaded')

  for seq_length in range(1, 3):
    f = open('/home/jhlim/data/seq.learn.%d.csv'%(seq_length), 'r')
    learn_instances = list(map(lambda x:x.replace('\n', '').split(','), f.readlines()))
    f.close()

    np.random.shuffle(learn_instances)

    learn_X = []; learn_Y = []

    for seq in learn_instances:
      sub_x = []

      try:
        flag = -1
        for element in seq[:-1]:
          flag += 1
          d_bertfeatures = d_bertfeatures1 if flag == 0 else d_bertfeatures2
          liwc_features = []

          #if d_w2vfeatures.has_key(element):
          if element in d_userfeatures and element in d_features:
            liwc_features = d_features[element]['liwc']
            bert_features = d_bertfeatures[element]

          else:
            continue

          if liwc_features != []:
              sub_x.append(np.array(bert_features + liwc_features))

        if (len(sub_x) == seq_length):
          learn_X.append(np.array(sub_x))
          learn_Y.append(seq[-1])

      except Exception as e:
        continue

    print (Counter(learn_Y))
    learn_X_reshape = np.reshape(np.array(learn_X), [-1, seq_length*input_dim])
    sample_model = RandomUnderSampler(random_state=42)
    learn_X, learn_Y = sample_model.fit_sample(learn_X_reshape, learn_Y)

    print (Counter(learn_Y))

    f = open('/home/jhlim/data/seq.test.%d.csv'%(seq_length), 'r')
    test_instances = list(map(lambda x:x.replace('\n', '').split(','), f.readlines()))
    f.close()

    np.random.shuffle(test_instances)

    test_X = []; test_Y = []

    for seq in test_instances:
      sub_x = []

      try:
        flag = -1
        for element in seq[:-1]:
          flag += 1
          d_bertfeatures = d_bertfeatures1 if flag == 0 else d_bertfeatures2
          liwc_features = []
          user_features = []

          if element in d_userfeatures and element in d_features:
            liwc_features = d_features[element]['liwc']
            user_features = d_userfeatures[element]['user']
            bert_features = d_bertfeatures[element]

          else:
              continue

          if liwc_features != []:
              if exclude_newbie == 1 and user_features == [0.0, 0.0]:
                  continue
              sub_x.append(np.array(bert_features + liwc_features))

        if (len(sub_x) == seq_length):
            test_X.append(np.array(sub_x))
            test_Y.append(seq[-1])

      except Exception as e:
        continue

    test_X = np.reshape(np.array(test_X), [-1, seq_length*input_dim])

    runRFModel(seq_length, learn_X, learn_Y, test_X, test_Y, input_dim)

    '''
    clf = RandomForestClassifier(n_jobs=-1)
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
    print ('seq_length: %d, n: %d, corrects: %d, acc: %.3f, auc_v: %.3f'%(
        seq_length, n, corrects, accuracy_score(test_Y, out), roc_auc_score(test_Y, out)))
    '''


if __name__ == '__main__':
  tf.app.run(main=main, argv=[sys.argv])

