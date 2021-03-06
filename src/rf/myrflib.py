import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import *

def runRFModel(seq_length, learn_X, learn_Y, test_X, test_Y, max_features=4):
    test_parent = False
    input_dim = int(len(learn_X[0]) / seq_length)
    print ('input_dim: %d'%(input_dim))

    # To test 1st comment -> 2nd comment
    if test_parent:
        learn_X = np.array(learn_X)[:, (seq_length-1)*input_dim:].tolist()
        test_X = np.array(test_X)[:, (seq_length-1)*input_dim:].tolist()
        print (np.array(learn_X).shape)

    max_features = 1 + input_dim // 2 

    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0, max_features=max_features)
    clf.fit(learn_X, learn_Y)

    print (np.array(learn_X).shape)
    print (np.array(test_X).shape)

    out = list(map(int, clf.predict(test_X).tolist()))
    print (clf.score(test_X, test_Y))
    predicts = []

    test_Y = [int(i) for i in test_Y]

    idx = []; i = 0
    for v1, v2 in zip(out, test_Y):
      decision = False

      if v1 == int(v2):
        decision = True
        idx.append(str(i))
      predicts.append(decision)
      i += 1


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
    print ('seq_length: %d, n: %d, corrects: %d, acc: %.3f, auc_v: %.3f, f1: '%(
        seq_length, n, corrects, accuracy_score(test_Y, out), roc_auc_score(test_Y, out)), precision_recall_fscore_support(test_Y, out)[2])
    print (classification_report(test_Y, out, labels=[0, 1]))
    f = open('result.txt', 'w')
    f.write('\n'.join(idx))
    f.close()


