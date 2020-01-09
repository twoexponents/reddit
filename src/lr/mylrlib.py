import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.metrics import classification_report

def runLRModel(seq_length, learn_X, learn_Y, test_X, test_Y):
    test_parent = False
    input_dim = int(len(learn_X[0]) / seq_length)
    print ('input_dim:', input_dim)

    # To test 1st comment -> 2nd comment
    if test_parent:
        learn_X = np.array(learn_X)[:, (seq_length-1)*input_dim:].tolist()
        test_X = np.array(test_X)[:, (seq_length-1)*input_dim:].tolist()
        print (np.array(learn_X).shape)

    clf = LogisticRegression(penalty='l2', random_state=0, solver='lbfgs')
    clf.fit(learn_X, learn_Y)

    out = list(map(int, clf.predict(test_X).tolist()))
    predicts = []

    test_Y = [int(i) for i in test_Y]

    for v1, v2 in zip(out, test_Y):
      decision = False

      if v1 == int(v2):
        decision = True
      predicts.append(decision)

    n = len(predicts)
    corrects = len(list(filter(lambda x:x, predicts)))

    #print ('%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f'%(
    #  seq_length, n, corrects, acc,
    #  prec[0], prec[1], rec[0], rec[1], f1[0], f1[1], auc_v, ap))
    print ('seq_length: %d, n: %d, corrects: %d, acc: %.3f, auc_v: %.3f: '%(
        seq_length, n, corrects, accuracy_score(test_Y, out), roc_auc_score(test_Y, out)))
    print (classification_report(test_Y, out, labels=[0, 1]))

