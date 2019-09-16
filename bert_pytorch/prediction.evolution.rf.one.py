import tensorflow as tf
import numpy as np
import sys
#import data_processing as dp
import cPickle as pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import pandas as pd
from pandas import DataFrame, Series

#from imblearn.under_sampling import RandomUnderSampler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample


from collections import Counter

#common_features_fields = ['vader_score', 'vader', 'difficulty']

#cont_features_fields = common_features_fields + post_features_fields
#cont_features_fields = post_features_fields
#cont_features_fields = common_features_fields

len_w2v_features = 300


#input_dim = len(cont_features_fields) + len(user_features_fields) + len_liwc_fetures + len_w2v_features
#input_dim = len(cont_features_fields) + len_liwc_fetures + len_w2v_features# number of features
#input_dim = len(cont_features_fields) + len_liwc_fetures
#input_dim = len(cont_features_fields) + len_w2v_features
input_dim = len_w2v_features
#input_dim = len_liwc_fetures
#input_dim = len(cont_features_fields)
#input_dim = len(user_features_fields)
#input_dim = len(user_features_fields) + len_liwc_fetures
#input_dim = len(cont_features_fields) + len_liwc_fetures + len(user_features_fields)
#input_dim = len(cont_features_fields) + len_liwc_fetures + len_w2v_features + len(user_features_fields)
#input_dim = len(cont_features_fields) + len(user_features_fields)

output_dim = 2 # (0, 1)
hidden_size = 500
learning_rate = 0.001
batch_size = 1000
epochs = 1


def main(argv):
  # 1.1 load feature dataset
  #d_features = pickle.load(open('../data/contentfeatures.others.p', 'r'))
  d_w2vfeatures = pickle.load(open('/home/jhlim/data/contentfeatures.googlenews.nozero.p', 'r'))
  #d_userfeatures = pickle.load(open('../data/userfeatures.activity.p', 'r'))

  print 'features are loaded'

  for seq_length in xrange(1, 6):
    input_length = 1
    f = open('/home/jhlim/data/seq.learn.%d.csv'%(seq_length), 'r')
    learn_instances = map(lambda x:x.replace('\n', '').split(','), f.readlines())
    f.close()

    np.random.shuffle(learn_instances)

    learn_X = []; learn_Y = []

    for seq in learn_instances:
      sub_x = []

      try:
        for element in seq[seq_length-1:-1]:
          #cont_features = [0.0]*len(cont_features_fields)
          #liwc_features = [0.0]*len_liwc_fetures
          w2v_features = [0.0]*len_w2v_features
          #user_features = [0.0]*len(user_features_fields)

          if d_w2vfeatures.has_key(element):
          #if d_userfeatures.has_key(element):
            #cont_features = d_features[element]['cont']
            #cont_features = d_features[element]['cont'][:len(common_features_fields)]
            #cont_features = d_features[element]['cont'][len(common_features_fields):]
            #liwc_features = d_features[element]['liwc']
            w2v_features = d_w2vfeatures[element]['google.mean']
            #user_features = d_userfeatures[element]['user'][0:2]

            #if len(cont_features) < len(cont_features_fields):
            #    cont_features += [0.0]*(len(cont_features_fields) - len(cont_features))
          else:
              continue

          #sub_x.append(np.array(cont_features))
          sub_x.append(np.array(w2v_features))
          #sub_x.append(np.array(cont_features+liwc_features+w2v_features.tolist()))
          #sub_x.append(np.array(cont_features+w2v_features.tolist()+user_features))
          #sub_x.append(np.array(cont_features+liwc_features+w2v_features.tolist()+user_features))
          #sub_x.append(np.array(w2v_features.tolist()))
          #sub_x.append(np.array(cont_features+liwc_features))
          #sub_x.append(np.array(user_features+liwc_features))
          #sub_x.append(np.array(user_features))
          #sub_x.append(np.array(cont_features+liwc_features+user_features))
          #sub_x.append(np.array(cont_features+liwc_features+w2v_features.tolist()+user_features))
          #sub_x.append(np.array(cont_features+user_features))

        if (len(sub_x) == input_length):
          learn_X.append(np.array(sub_x))
          learn_Y.append(seq[-1])

      except Exception, e:
        # print e
        continue

    print Counter(learn_Y)

    learn_X_reshape = np.reshape(np.array(learn_X), [-1, input_length*input_dim])
    df = DataFrame(learn_X_reshape)
    df2 = DataFrame(learn_Y)

    df_class1 = df[df.iloc[:, -1] == '0']
    df_class2 = df[df.iloc[:, -1] == '1']

    if len(df_class1) > len(df_class2):
      df_majority = df_class1
      df_minority = df_class2
    else:
      df_majority = df_class2
      df_minority = df_class1

    df_majority_downsampled = resample(df_majority,
                                    replace=False,
                                    n_samples=len(df_minority),
                                    random_state=123)
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])

    df = df_downsampled

    learn_X = df.iloc[:, :-1].as_matrix()
    learn_Y = df.iloc[:, -1].as_matrix()
    
    #sample_model = RandomUnderSampler(random_state=42)
    #learn_X, learn_Y = sample_model.fit_sample(learn_X_reshape, learn_Y)
    #learn_X = np.reshape(learn_X, [-1, seq_length, input_dim])

    print Counter(learn_Y)

    f = open('/home/jhlim/data/seq.test.%d.csv'%(seq_length), 'r')
    test_instances = map(lambda x:x.replace('\n', '').split(','), f.readlines())
    f.close()

    np.random.shuffle(test_instances)

    test_X = []; test_Y = []

    for seq in test_instances:
      sub_x = []

      try:
        for element in seq[seq_length-1:-1]:
          #cont_features = [0.0]*len(cont_features_fields)
          #liwc_features = [0.0]*len_liwc_fetures
          w2v_features = [0.0]*len_w2v_features
          #user_features = [0.0]*len(user_features_fields)

          if d_features.has_key(element):
          #if d_userfeatures.has_key(element):
            #cont_features = d_features[element]['cont']
            #cont_features = d_features[element]['cont'][:len(common_features_fields)]
            #cont_features = d_features[element]['cont'][len(common_features_fields):]
            #liwc_features = d_features[element]['liwc']
            w2v_features = d_w2vfeatures[element]['google.mean']
            #user_features = d_userfeatures[element]['user'][0:2]

            #if len(cont_features) < len(cont_features_fields):
            #    cont_features += [0.0]*(len(cont_features_fields) - len(cont_features))
          else:
              continue

          #sub_x.append(np.array(cont_features))
          sub_x.append(np.array(w2v_features))
          #sub_x.append(np.array(cont_features+liwc_features+w2v_features.tolist()))
          #sub_x.append(np.array(cont_features+liwc_features))
          #sub_x.append(np.array(cont_features+w2v_features.tolist()+user_features))
          #sub_x.append(np.array(cont_features+liwc_features+w2v_features.tolist()+user_features))
          #sub_x.append(np.array(w2v_features.tolist()))
          #sub_x.append(np.array(cont_features+liwc_features))
          #sub_x.append(np.array(user_features+liwc_features))
          #sub_x.append(np.array(user_features))
          #sub_x.append(np.array(cont_features+liwc_features+user_features))
          #sub_x.append(np.array(cont_features+liwc_features+w2v_features.tolist()+user_features))
          #sub_x.append(np.array(cont_features+user_features))

        if (len(sub_x) == input_length):
            test_X.append(np.array(sub_x))
            test_Y.append(seq[-1])

      except Exception, e:
        continue

    test_X = np.reshape(np.array(test_X), [-1, input_length*input_dim])
    #test_X = np.reshape(test_X, [-1, seq_length, input_dim])


    clf = RandomForestClassifier(n_jobs=-1)
    clf.fit(learn_X, learn_Y)

    out = map(int, clf.predict(test_X).tolist())
    predicts = []

    for v1, v2 in zip(out, test_Y):
      decision = False

      if v1 == int(v2):
        decision = True
      predicts.append(decision)

    y_true = map(int, test_Y)

    n = len(predicts)
    corrects = len(filter(lambda x:x, predicts))

    acc = float(corrects)/n
    prec, rec, f1, support, = precision_recall_fscore_support(y_true, out)
    fpr, tpr, thresholds = roc_curve(y_true, out)
    auc_v = auc(fpr, tpr)
    #ap = average_precision_score (y_true, out)
    #print '%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f'%(
    #  seq_length, n, corrects, acc,
    #  prec[0], prec[1], rec[0], rec[1], f1[0], f1[1], auc_v, ap)
    print 'seq_length: %d, n: %d, corrects: %d, acc: %.3f, auc_v: %.3f'%(
      seq_length, n, corrects, acc, auc_v)

    #print seq_length, n, corrects
    #print prec, rec, f1, support, auc, ap



if __name__ == '__main__':
  tf.app.run(main=main, argv=[sys.argv])

