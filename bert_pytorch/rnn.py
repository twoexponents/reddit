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
import sys
from twoLayerNet import TwoLayerNet
from sklearn.utils import resample
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score

user_features_fields = ['posts', 'comments']
len_liwc_features = 29
input_dim1, input_dim2, input_dim3 = 128, len(user_features_fields) + len_liwc_features, 2
hidden_size2, hidden_size3 = 32, 4
batch_size1, batch_size2, batch_size3 = 32, 100, 10
epochs1, epochs2, epochs3 = 1, 1, 1
MAX_LEN = input_dim1

with open('/home/jhlim/data/userfeatures.activity.p', 'rb') as f:
    d_userfeatures = pickle.load(f)
with open('/home/jhlim/data/contentfeatures.others.p', 'rb') as f:
    d_liwcfeatures = pickle.load(f)

if torch.cuda.is_available():
    print ('cuda is available. use gpu.')
    device = torch.device("cuda")
else:
    print ('cuda is not available. use cpu.')
    device = torch.device("cpu")

if len(sys.argv) <= 1:
    input_length = 1
else:
    input_length = int(sys.argv[1])

# START
for seq_length in range(input_length, input_length+1):
    print ('seq_length: %d'%(seq_length))
    train_set = "data/leaf/seq.learn." + str(seq_length) + ".tsv"
    test_set = "data/leaf/seq.test." + str(seq_length) + ".tsv"

    sources = ['source' + str(i) for i in range(seq_length)]
    sentences = ['sentence' + str(i) for i in range(seq_length)]
    df = pd.read_csv(train_set, delimiter='\t', header=None, engine='python', 
        names=sources
            + ['label', 'label_notes']
            + sentences)

    df['sentence_source'] = df[sources].values.tolist()
    df['sentence'] = df[sentences].values.tolist()

    df = df.drop(columns=sources + sentences)

    # jhlim: additional features dataset
    learn_instances = df.sentence_source.values

    extra_features = []
    for seq in learn_instances:
        sub_x = []
        for element in seq: # for each source id
            if element in d_userfeatures and element in d_liwcfeatures:
                user_features = d_userfeatures[element]['user'][0:2]
                liwc_features = d_liwcfeatures[element]['liwc'][0:29]
                sub_x.append(user_features + liwc_features)
            else:
                sub_x = None
                break
        extra_features.append(sub_x)

    df['extra'] = extra_features
    df = df.dropna()

    print (df.extra.sample(10))

    df = mylib.processDataFrame(df, is_training=True)

    
    input_ids, attention_masks, labels = mylib.makeBertElementsList(df, MAX_LEN)

    extras = df.extra.values

    # Use train_test_split to split our data into train and validation sets for training
    train_inputs, validation_inputs, train_labels, validation_labels, train_extras, validation_extras = train_test_split(input_ids, labels, extras, random_state=2018, test_size=0.1)
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                 random_state=2018, test_size=0.1)
    train_extras = [item for item in train_extras]
    validation_extras = [item for item in validation_extras]
        
    # Convert all of our data into torch tensors, the required datatype for our model
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)
    train_extras = torch.tensor(train_extras)
    validation_extras = torch.tensor(validation_extras)

    train_data = TensorDataset(train_inputs, train_masks, train_labels, train_extras)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size1)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels, validation_extras)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size1)

    #model1 = BertForSequenceClassification.from_pretrained("bert-base-uncased", output_hidden_states=True)
    model1 = BertForSequenceClassification.from_pretrained("./models/len" + str(seq_length) + '/')
    model1.eval()

    N, D_in, H, D_out = batch_size2, input_dim2, hidden_size2, 2
    model2 = LSTM(D_in, H, D_out)
    optimizer2 = AdamW(model2.parameters(), lr=0.005)
    
    if torch.cuda.is_available():
        model1.cuda()
        model2.cuda()

    model2.train()

    # Model 2: LSTM Training
    print ('Start model 2 LSTM Training')
    for _ in trange(epochs3, desc="Epoch"):
        tr_loss = 0; nb_tr_steps = 0

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels, b_extras = batch

            for i in range(seq_length):
                optimizer2.zero_grad()
                with torch.no_grad():
                    outputs = model1(b_input_ids[:][i], token_type_ids=None, attention_mask=b_input_mask[:][i], labels=b_labels[:][i])
                    loss, logits = outputs[:2]

            with torch.no_grad():
                outputs = model1(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                loss, logits = outputs[:2]            
                logits = logits.to('cpu').numpy()
                output1 = np.argmax(logits, axis=1)
                output2 = model2(b_extras)
                output2 = output2.to('cpu').numpy()
                output2 = np.argmax(logits, axis=1)
            output1 = torch.tensor(output1)
            output1 = output1.to(device)
            output2 = torch.tensor(output2)
            output2 = output2.to(device)
            output3 = model3(torch.cat((output1, output2), dim=1))
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(output3, b_labels)
            loss.backward()
            optimizer3.step()

            tr_loss += loss.item()
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss/nb_tr_steps))

        model3.eval()

        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels, b_extras = batch
            with torch.no_grad():
                input = b_extras
                predicts = model2(input)

            predicts = predicts.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_accuracy = mylib.flat_accuracy(predicts, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        print ("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

    # Make a Testset

    df = pd.read_csv(test_set, delimiter='\t', header=None, engine='python', names=['sentence_source', 'label', 'label_notes', 'sentence'])

    element_list = df.sentence_source.values
    extra_features = []
    for element in element_list:
        if element in d_userfeatures and element in d_liwcfeatures:
            user_features = d_userfeatures[element]['user'][0:2]
            liwc_features = d_liwcfeatures[element]['liwc'][0:29]
            extra_features.append(user_features + liwc_features)
        else:
            # user_features = [0.0] * 2
            # liwc_features = [0.0] * 29
            extra_features.append(None)

    df['extra'] = extra_features
    df = df.dropna()
    df = mylib.processDataFrame(df, is_training=False)
    input_ids, attention_masks, labels = mylib.makeBertElements(df, MAX_LEN)

    extras = df.extra.values
    extras = [item for item in extras]

    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)
    prediction_labels = torch.tensor(labels)
    prediction_extras = torch.tensor(extras)
      
    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels, prediction_extras)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size3)

    # Prediction on test set

    # Put model in evaluation mode
    model3.eval()

    # Tracking variables
    predictions , true_labels = [], []

    print ('Start a prediction')
    # Predict
    for batch in prediction_dataloader:
      batch = tuple(t.to(device) for t in batch)
      b_input_ids, b_input_mask, b_labels, b_extras = batch
      # Telling the model not to compute or store gradients, saving memory and speeding up prediction
      with torch.no_grad():
        outputs = model1(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss, logits = outputs[:2]
        logits = logits.to('cpu').numpy()
        output1 = np.argmax(logits, axis=1)
        b_extras = b_extras.float()
        output2 = model2(b_extras)
        output2 = output2.to('cpu').numpy()
        output2 = np.argmax(output2, axis=1)
        output3 = model3(torch.cat((output1, output2), dim=1))

      # Move logits and labels to CPU
      output3 = output3.detach().cpu().numpy()
      label_ids = b_labels.to('cpu').numpy()
      
      # Store predictions and true labels
      predictions.append(output3)
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

    del model1, model2, model3, outputs, output1, output2, output3
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

