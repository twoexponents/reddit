import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import pandas as pd
import numpy as np
import sys
import pickle
import time
from twoLayerNet import TwoLayerNet
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score

from collections import Counter
from operator import itemgetter

len_liwc_features = 93
len_w2v_features = 300

input_dim = len_liwc_features

output_dim = 2 # (range 0 to 1)
hidden_size = 200
learning_rate = 0.01
batch_size = 100
epochs = 100

def main(argv):
    start_time = time.time()
    input_length = 1 if len(sys.argv) <= 1 else int(sys.argv[1])
    print ('sequence len: %d'%(input_length))
    print ('learning_rate: %f, batch_size %d, epochs %d' %(learning_rate, batch_size, epochs))

    # 1.1 load feature dataset
    with open('/home/jhlim/data/contentfeatures.others.p', 'rb') as f:
        d_features = pickle.load(f)

    print ('features are loaded')

    for seq_length in range(input_length, input_length+1):
        f = open('/home/jhlim/data/seq.learn.%d.csv'%(seq_length), 'r')
        learn_instances = list(map(lambda x:x.replace('\n', '').split(','), f.readlines()))
        f.close()

        np.random.shuffle(learn_instances)

        print (len(d_features))

        learn_X = []; learn_Y = []
        for seq in learn_instances:
            sub_x = []

            try:
                for element in seq[:-1]: # seq[-1] : Y. element: 't3_7dfvv'
                    liwc_features = []
                    if element in d_features:
                        liwc_features = d_features[element]['liwc']
                    else:
                        continue
                    if liwc_features != []:
                        sub_x.append(np.array(liwc_features))

                if (len(sub_x) == seq_length):
                    learn_X.append(np.array(sub_x)) # feature list
                    learn_Y.append(float(seq[-1]))

            except Exception as e:
                # print e
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

        f = open('/home/jhlim/data/seq.test.%d.csv'%(seq_length), 'r')
        test_instances = list(map(lambda x:x.replace('\n', '').split(','), f.readlines()))
        f.close()

        np.random.shuffle(test_instances)

        test_X = []; test_Y = []

        for seq in test_instances:
            sub_x = []

            try:
                for element in seq[:-1]:
                    liwc_features = []
                    if element in d_features:
                        liwc_features = d_features[element]['liwc']
                    else:
                        continue
                    if liwc_features != []:
                        sub_x.append(np.array(liwc_features))
                if (len(sub_x) == seq_length):
                    test_X.append(np.array(sub_x))
                    test_Y.append(float(seq[-1]))
            except Exception as e:
                continue

        test_Y = list(map(lambda x:[x], test_Y))
        
        print ('Data loading Complete learn:%d, test:%d'%(len(learn_Y), len(test_Y)))

        # 2. Run RNN

        train_inputs = learn_X
        train_labels = learn_Y
        train_inputs = torch.tensor(train_inputs) 
        train_labels = torch.tensor(train_labels)

        train_data = TensorDataset(train_inputs, train_labels)
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        prediction_inputs = test_X
        prediction_labels = test_Y
        prediction_inputs = torch.tensor(prediction_inputs)
        prediction_labels = torch.tensor(prediction_labels)

        prediction_data = TensorDataset(prediction_inputs, prediction_labels)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

        # TwoLayerNet
        N, D_in, H, D_out = batch_size, input_dim, hidden_size, output_dim
        model = TwoLayerNet(D_in, H, D_out)
        model = model.double()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)

        for e in range(epochs):
            # Training

            tr_loss = 0
            nb_tr_steps = 0

            model.train()

            for step, batch in enumerate(train_dataloader):
                batch = tuple(t for t in batch)
                b_inputs, b_labels = batch
                optimizer.zero_grad()
                b_inputs = b_inputs.double()
                print (b_inputs)
                output = model(b_inputs)
                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(output, b_labels)
                loss.backward()
                optimizer.step()

                # Update tracking variables
                tr_loss += loss.item()
                nb_tr_steps += 1

            print ('Train loss: {}'.format(tr_loss/nb_tr_steps))

            # Tracking variables
            predictions, true_labels = [], []

            # Predict
            if (e % 5 == 0 and e != 0):
                model.eval()

                for batch in prediction_dataloader:
                    batch = tuple(t for t in batch)
                    b_inputs, b_labels = batch
                    with torch.no_grad():
                        predicts = model(b_input)

                    predictions.append(predicts)
                    true_labels.append(label_ids)

                flat_predictions = [item for sublist in predictions for item in sublist]
                flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
                flat_true_labels = [item for sublist in true_labels for item in sublist]

                predicts = []
                for v1, v2 in zip(flat_true_labels, flat_predictions):
                    decision = True if v1 == v2 else False
                    predicts.append(decision)

                print ('seq_length: %d, # predicts: %d, # corrects %d, acc: %.3f, auc: %.3f'%
                    (seq_length, len(predicts), len(corrects), accuracy_score(flat_true_labels, flat_predictions), roc_auc_score(flat_true_labels, flat_predictions)))

            print ('work time: %s sec\n'%(time.time()-start_time))


if __name__ == '__main__':
    main(sys.argv)

