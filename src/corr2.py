import pandas as pd
import numpy as np
import sys
import pickle
import math

from collections import Counter
from operator import itemgetter
pd.options.display.max_rows = 1000

common_features_fields = ['vader_score', 'vader', 'difficulty']
user_features_fields = ['posts', 'comments', 'receives']
cont_features_fields = common_features_fields
len_liwc_features = 93
len_bert_features = 768

d_userfeatures = pickle.load(open('/home/jhlim/data/userfeatures.activity.p', 'rb')) #3
d_features = pickle.load(open('/home/jhlim/data/contentfeatures.others.p', 'rb')) #93 + 3
d_timefeatures = pickle.load(open('/home/jhlim/data/temporalfeatures.p', 'rb')) #1
d_bertfeatures = pickle.load(open('/home/jhlim/data/bertfeatures1.p', 'rb')) #768

seq_length = int(sys.argv[1])
print ('seq_length: ' + str(seq_length))

f = open('/home/jhlim/data/seq.learn.' + str(seq_length) + '.csv', 'r')
learn_instances = list(map(lambda x:x.replace('\n', '').split(','), f.readlines()))
f.close()

np.random.shuffle(learn_instances)

learn_X = []; learn_Y = []
for seq in learn_instances:
    sub_x = []

    try:
        for element in seq[seq_length-1:seq_length]: # seq[-1] : Y. element: 't3_7dfvv'
            user_features = []
            liwc_features = []
            cont_features = []
            time_features = []
            bert_features = []
            if element in d_userfeatures and element in d_features: #and element in d_bertfeatures:
                user_features = d_userfeatures[element]['user']
                liwc_features = d_features[element]['liwc']
                cont_features = d_features[element]['cont'][:3]
                time_features = d_timefeatures[element]['ict']
                bert_features = d_bertfeatures[element]
            else:
                continue
            
            #if user_features != [] and liwc_features != [] and cont_features != [] and bert_features != []:
            if user_features != []:
                #sub_x.append((user_features + liwc_features + cont_features + time_features))
                sub_x.append((user_features + liwc_features + cont_features + time_features + bert_features))

        if (len(sub_x) == 1):
            learn_X.append(np.array(sub_x))
            learn_Y.append(int(seq[-1]))

    except Exception as e:
        continue

x = np.array(learn_X)
print (x.shape)
print (x[10, :, :])
user_f = x[:, 0, 0:len(user_features_fields)]
liwc_f = x[:, 0, len(user_features_fields):len(user_features_fields) + len_liwc_features]
cont_f = x[:, 0, len(user_features_fields)+len_liwc_features:len(user_features_fields) + len_liwc_features + len(cont_features_fields)]
time_f = x[:, 0, len(user_features_fields)+len_liwc_features + len(cont_features_fields) : len(user_features_fields) + len_liwc_features + len(cont_features_fields) + 1]
bert_f = x[:, 0, len(user_features_fields) + len_liwc_features + len(cont_features_fields) + 1 :]

#print (np.array(bert_f).shape)


df = pd.DataFrame()
df['posts'] = user_f[:, 0]
df['comments'] = user_f[:, 1]
df['receives'] = user_f[:, 2]
for i in range(len_liwc_features):
    df['liwc' + str(i)] = liwc_f[:, i:i+1]
for i in range(len(cont_features_fields)):
    df['cont' + str(i)] = cont_f[:, i:i+1]
df['time'] = time_f[:, 0]
for i in range(len_bert_features):
    df['bert' + str(i)] = bert_f[:, i:i+1]

df_label = pd.DataFrame()
df_label['label'] = learn_Y

corr = df.corrwith(df_label['label'], method='pearson')
df2 = pd.DataFrame(corr, columns=['label'])

co = []
i = 0
for idx, value in corr.items():
    if math.pow(value, 2) >= 0.09:
        co.append('0.3')
    elif math.pow(value, 2) >= 0.04:
        co.append('0.2')
    elif math.pow(value, 2) >= 0.0225:
        co.append('0.15')
    else:
        co.append(' ')
    i += 1

df2['corr'] = co

print (df2.corr)








