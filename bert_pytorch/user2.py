import tensorflow as tf
import torch
from pytorch_transformers import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle

from sklearn.utils import resample
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
from twoLayerNet import TwoLayerNet

user_features_fields = ['posts', 'comments']
input_dim = len(user_features_fields)
with open('/home/jhlim/SequencePrediction/data/userfeatures.activity.p', 'rb') as f:
    d_userfeatures = pickle.load(f)

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

if torch.cuda.is_available():
    print ('cuda is available. use gpu.')
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    torch.cuda.get_device_name(0)
else:
    print ('cuda is not available. use cpu.')
    device = torch.device("cpu")

# START
for seq_length in range(1, 6):
    print ('seq_length: %d'%(seq_length))
    train_set = "data/leaf_depth/seq.learn." + str(seq_length) + ".tsv"
    test_set = "data/leaf_depth/seq.test." + str(seq_length) + ".tsv"

    df = pd.read_csv(train_set, delimiter='\t', header=None, engine='python', names=['sentence_source', 'label', 'label_notes', 'sentence'])
    df = df.dropna()
    df.label = df.label.astype(int) 


    # jhlim: additional features dataset
    element_list = df.sentence_source.values
    extra_features = []
    notin = 0
    for element in element_list:
        user_features = [0.0]*len(user_features_fields)
        if element in d_userfeatures:
            user_features = [float(item) for item in d_userfeatures[element]['user'][0:2]]
        else:
            user_features = -1.0
        extra_features.append(user_features)
    
    df['user'] = extra_features
    df = df[df.user != -1.0]

    # jhlim: undersampling training set using dataframe
    df_class1 = df[df.label == 0]
    df_class2 = df[df.label == 1]

    (df_majority, df_minority) = (df_class1, df_class2) if len(df_class1) > len(df_class2) else (df_class2, df_class1)

    print ("train dataset [%d]: %d, [%d]: %d"%(df_majority.label.values[0], len(df_majority), df_minority.label.values[0], len(df_minority)))
    length_minority = 30000 if len(df_minority) > 30000 else len(df_minority)
    
    df_majority_downsampled = resample(df_majority, replace=False, n_samples=length_minority, random_state=123)
    df_minority_downsampled = resample(df_minority, replace=False, n_samples=length_minority, random_state=123)
    df_downsampled = pd.concat([df_majority_downsampled, df_minority_downsampled])
    df = df_downsampled

    print ("train dataset [%d]: %d, [%d]: %d"%(df_majority_downsampled.label.values[0], len(df_majority_downsampled), df_minority_downsampled.label.values[0], len(df_minority_downsampled)))

    extras = df.user.values
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


    # Select a batch size for training. For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32
    batch_size = 100

    # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
    # with an iterator the entire dataset does not need to be loaded into memory

    train_data = TensorDataset(train_labels, train_extras)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_labels, validation_extras)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    # TwoLayerNet
    N, D_in, H, D_out = batch_size, 2, 100, 2
    model = TwoLayerNet(D_in, H, D_out)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)

    # Store our loss and accuracy for plotting
    train_loss_set = []

    # Number of training epochs (authors recommend between 2 and 4)
    epochs = 50

    # trange is a tqdm wrapper around the normal python range
    for _ in trange(epochs, desc="Epoch"):
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

        print("Train loss: {}".format(tr_loss/nb_tr_steps))

        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Tracking variables 
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
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

            tmp_eval_accuracy = flat_accuracy(predicts, label_ids)

            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

    df = pd.read_csv(test_set, delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'], engine='python')

    df = df.dropna()
    df.label = df.label.astype(int)

    # jhlim: additional features dataset
    element_list = df.sentence_source.values
    extra_features = []
    for element in element_list:
        user_features = [0.0]*len(user_features_fields)
        if element in d_userfeatures:
            user_features = [float(item) for item in d_userfeatures[element]['user'][0:2]]
        else:
            user_features = -1.0
        extra_features.append(user_features)

    df['user'] = extra_features
    df = df[df.user != -1.0]

    df_class1 = df[df.label == 0]
    df_class2 = df[df.label == 1]

    print ("test dataset [%d]: %d, [%d]: %d"%(df_class1.label.values[0], len(df_class1), df_class2.label.values[0], len(df_class2)))

    extras = df.user.values
    labels = df.label.values

    prediction_labels = torch.tensor(labels)
    prediction_extras = torch.tensor(list(extras))
      
    prediction_data = TensorDataset(prediction_labels, prediction_extras)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    # Prediction on test set

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
        decision = False

        if v1 == v2:
            decision = True
        predicts.append(decision)

    num_predicts = len(predicts)
    num_corrects = len(list(filter(lambda x:x, predicts)))

    fpr, tpr, thresholds = roc_curve(list(map(int, flat_true_labels)), flat_predictions)
    print ('# predicts: %d, # corrects: %d, # 0: %d, # 1: %d, acc: %f, auc: %f'%
        (num_predicts, num_corrects, len(list(filter(lambda x:x == 0, flat_predictions))), len(list(filter(lambda x:x == 1, flat_predictions))), num_corrects/num_predicts, auc(fpr,tpr)))


