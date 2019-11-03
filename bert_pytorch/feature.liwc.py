import tensorflow as tf
import torch
from pytorch_transformers import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
import pandas as pd
import numpy as np
import pickle
import mylib
from twoLayerNet import TwoLayerNet
from sklearn.utils import resample
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score

len_liwc_features = 29 
common_features_fields = ['vader_score', 'vader', 'difficulty']
cont_features_fields = common_features_fields
input_dim = len_liwc_features
MAX_LEN = 128
hidden_size = 20
learning_rate = 0.01
batch_size = 32
epochs = 100

with open('/home/jhlim/data/contentfeatures.others.p', 'rb') as f:
    d_features = pickle.load(f)

if torch.cuda.is_available():
    print ('cuda is available. use gpu.')
    device = torch.device("cuda")
else:
    print ('cuda is not available. use cpu.')
    device = torch.device("cpu")


# START
for seq_length in range(1, 6):
    print ('seq_length: %d'%(seq_length))
    train_set = "data/leaf_depth/seq.learn." + str(seq_length) + ".tsv"
    test_set = "data/leaf_depth/seq.test." + str(seq_length) + ".tsv"

    df = pd.read_csv(train_set, delimiter='\t', header=None, engine='python', names=['sentence_source', 'label', 'label_notes', 'sentence'])


    # jhlim: additional features dataset
    element_list = df.sentence_source.values
    extra_features = []
    for element in element_list:
        if element in d_features:
            liwc_features = [float(item) for item in d_features[element]['liwc'][0:29]]
        else:
            liwc_features = None
        extra_features.append(liwc_features)
    print (len(df)) 
    df['liwc'] = extra_features
    df = df.dropna()
    print (len(df))

    df = mylib.processDataFrame(df, is_training=True)
    input_ids, attention_masks, labels = mylib.makeBertElements(df, MAX_LEN)

    extras = df.liwc.values
    labels = df.label.values
    # Use train_test_split to split our data into train and validation sets for training
    train_labels, validation_labels, train_extras, validation_extras = train_test_split(labels, extras, random_state=2018, test_size=0.1)

    train_extras = list(train_extras)
    validation_extras = list(validation_extras)

    # Convert all of our data into torch tensors, the required datatype for our model
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    train_extras = torch.tensor(train_extras)
    validation_extras = torch.tensor(validation_extras)

    train_data = TensorDataset(train_labels, train_extras)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_labels, validation_extras)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    df = pd.read_csv(test_set, delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'], engine='python')

    #df = df.dropna()
    #df.label = df.label.astype(int)

    # jhlim: additional features dataset
    element_list = df.sentence_source.values
    extra_features = []
    for element in element_list:
        if element in d_features:
            liwc_features = [float(item) for item in d_features[element]['liwc'][0:29]]
        else:
            liwc_features = None
        extra_features.append(liwc_features)

    df['liwc'] = extra_features
    df = df.dropna()

    df = mylib.processDataFrame(df, is_training=False) # Undersampling
    input_ids, attention_masks, labels = mylib.makeBertElements(df, MAX_LEN)

    extras = df.liwc.values
    labels = df.label.values

    prediction_labels = torch.tensor(labels)
    prediction_extras = torch.tensor(list(extras))
      
    prediction_data = TensorDataset(prediction_labels, prediction_extras)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    # Prediction on test set

    # TwoLayerNet
    N, D_in, H, D_out = batch_size, input_dim, hidden_size, 2
    model = TwoLayerNet(D_in, H, D_out)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Store our loss and accuracy for plotting
    train_loss_set = []

    # trange is a tqdm wrapper around the normal python range
    for e in trange(epochs, desc="Epoch"):
        # Training
      
        # Tracking variables
        tr_loss = 0
        nb_tr_steps = 0

        if torch.cuda.is_available():
            model.cuda()

        model.train()

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_labels, b_extras = batch
            optimizer.zero_grad()
            input = b_extras

            output = model(input)
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(output, b_labels)
            loss.backward()
            optimizer.step()

            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_steps += 1


        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Tracking variables 
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        predictions = []; true_labels = []
        # Evaluate data for one epoch
        for batch in prediction_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_labels, b_extras = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                input = b_extras
                predicts = model(input)

            # Move logits and labels to CPU
            predicts = predicts.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            predictions.append(predicts)
            true_labels.append(label_ids)

            #tmp_eval_accuracy = mylib.flat_accuracy(predicts, label_ids)

            #eval_accuracy += tmp_eval_accuracy
            #nb_eval_steps += 1
        flat_predictions = [item for sublist in predictions for item in sublist]
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
        flat_true_labels = [item for sublist in true_labels for item in sublist]
        
        if (e % 10 == 0): 
            print("Train loss: {}".format(tr_loss/nb_tr_steps))
            #print("Prediction Accuracy: {}".format(eval_accuracy/nb_eval_steps))
            print("Prediction AUC: {}".format(roc_auc_score(flat_true_labels, flat_predictions)))

    '''
    # Put model in evaluation mode
    model.eval()

    # Tracking variables
    predictions , true_labels = [], []

    # Predict
    for batch in prediction_dataloader:
      batch = tuple(t.to(device) for t in batch)
      b_labels, b_extras = batch
      with torch.no_grad():
        input = b_extras
        predicts = model(input)

      # Move logits and labels to CPU
      predicts = predicts.detach().cpu().numpy()
      label_ids = b_labels.to('cpu').numpy()
      
      # Store predictions and true labels
      predictions.append(predicts)
      true_labels.append(label_ids)

    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    flat_true_labels = [item for sublist in true_labels for item in sublist]

    predicts = []
    for v1, v2 in zip(flat_true_labels, flat_predictions):
        decision = True if v1 == v2 else False
        predicts.append(decision)

    print ('# predicts: %d, # corrects: %d, # 0: %d, # 1: %d, acc: %f, auc: %f'%
        (len(predicts), len(list(filter(lambda x:x, predicts))), len(list(filter(lambda x:x == 0, flat_predictions))), len(list(filter(lambda x:x == 1, flat_predictions))), accuracy_score(flat_true_labels, flat_predictions), roc_auc_score(flat_true_labels, flat_predictions)))

    '''


