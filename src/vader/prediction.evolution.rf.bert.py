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

common_features_fields = ['vader_score', 'vader', 'difficulty']
post_features_fields = ['pub_1h', 'pub_hd', 'pub_1d', 'max_similarity_1h', 
  'max_similarity_hd', 'max_similarity_1d', 'pub_time_0', 'pub_time_1',
  'pub_time_2', 'pub_time_3', 'pub_time_4', 'pub_time_5', 'pub_time_6',
  'pub_time_7', 'pub_time_8', 'pub_time_9', 'pub_time_10', 'pub_time_11',
  'pub_time_12', 'pub_time_13', 'pub_time_14', 'pub_time_15', 'pub_time_16',
  'pub_time_17', 'pub_time_18', 'pub_time_19', 'pub_time_20', 'pub_time_21',
  'pub_time_22', 'pub_time_23']
comment_features_fields = ['similarity_post', 'similarity_parent', 'inter_comment_time', 'prev_comments']
user_features_fields = ['posts', 'comments']

len_liwc_features = 93
len_w2v_features = 300
len_bert_features = 768

#input_dim = len_w2v_features
#input_dim = len_liwc_features
#input_dim = len(user_features_fields)
input_dim = len_bert_features

output_dim = 3 # (0, 1)


def main(argv):
  exclude_newbie = 0; input_length = 1
  if len(sys.argv) >= 3:
    exclude_newbie = int(sys.argv[2])
  if len(sys.argv) >= 2:
    input_length = int(sys.argv[1])
  print ('sequence len: %d' % (input_length))
  print ('exclude_newbie: %d'%(exclude_newbie))

  # 1.1 load feature dataset
  #d_features = pickle.load(open('../data/contentfeatures.others.p', 'r'))
  #d_w2vfeatures = pickle.load(open('../data/contentfeatures.googlenews.p', 'r'))


  d_userfeatures = pickle.load(open('/home/jhlim/data/userfeatures.activity.p', 'rb'))
  d_bertfeatures = pickle.load(open('/home/jhlim/data/bertfeatures2.p', 'rb'))

  print ('features are loaded')

  for seq_length in range(1, 3):
    f = open('/home/jhlim/data/vader/seq.learn.%d.csv'%(seq_length), 'r')
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
          #cont_features = [0.0]*len(cont_features_fields)
          #liwc_features = []
          #liwc_features = [0.0]*len_liwc_features
          bert_features = []

          #if d_w2vfeatures.has_key(element):
          if element in d_bertfeatures:
            #liwc_features = d_features[element]['liwc']
            #w2v_features = d_w2vfeatures[element]['google.mean'][0]
            bert_features = d_bertfeatures[element]

          else:
            continue

          if bert_features != []:
              sub_x.append(np.array(bert_features))

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

    f = open('/home/jhlim/data/vader/seq.test.%d.csv'%(seq_length), 'r')
    test_instances = list(map(lambda x:x.replace('\n', '').split(','), f.readlines()))
    f.close()

    np.random.shuffle(test_instances)

    test_X = []; test_Y = []

    for seq in test_instances:
      sub_x = []

      try:
        for element in seq[:-1]:
          #liwc_features = [0.0]*len_liwc_features
          bert_features = []
          user_features = []

          if element in d_bertfeatures:
            #liwc_features = d_features[element]['liwc']
            user_features = d_userfeatures[element]['user']
            bert_features = d_bertfeatures[element]

          else:
              continue

          if bert_features != []:
            if exclude_newbie == 1 and user_features == [0.0, 0.0]:
                continue
            sub_x.append(np.array(bert_features))

        if (len(sub_x) == seq_length):
            test_X.append(np.array(sub_x))
            test_Y.append(seq[-1])

      except Exception as e:
        continue

    test_X = np.reshape(np.array(test_X), [-1, seq_length*input_dim])

    runRFModel(seq_length, learn_X, learn_Y, test_X, test_Y, input_dim, output_dim)
    
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

