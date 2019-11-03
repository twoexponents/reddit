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
MAX_LEN = 128 # input_dim1
input_dim2 = len(user_features_fields) + len_liwc_features
input_dim3 = 2

hidden_size2 = 32
hidden_size3 = 2
batch_size1 = 32
batch_size2 = 100
batch_size3 = 10
epochs1 = 4
epochs2 = 20
epochs3 = 10

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


# START
for seq_length in range(1, 2):
    print ('seq_length: %d'%(seq_length))
    train_set = "data/leaf_depth/seq.learn." + str(seq_length) + ".tsv"
    test_set = "data/leaf_depth/seq.test." + str(seq_length) + ".tsv"

    df = pd.read_csv(train_set, delimiter='\t', header=None, engine='python', names=['sentence_source', 'label', 'label_notes', 'sentence'])

    # jhlim: additional features dataset
    element_list = df.sentence_source.values
    extra_features = []
    for element in element_list:
        if element in d_userfeatures and element in d_liwcfeatures:
            user_features = d_userfeatures[element]['user'][0:2]
            liwc_features = d_liwcfeatures[element]['liwc'][0:29]
            extra_features.append(user_features + liwc_features)
        else:
            extra_features.append(None)

    df['extra'] = extra_features
    df = df.dropna()


    df = mylib.processDataFrame(df, is_training=True)
    input_ids, attention_masks, labels = mylib.makeBertElements(df, MAX_LEN)

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

    try:
        model1 = BertForSequenceClassification.from_pretrained('./models/len' + str(seq_length) + '/') # load
        exist = True
    except Exception as e:
        exist = False
        

    if not exist:
        print ('failed to load exist pretrained model')
        # Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top. 
        model1 = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

        # Load model parameters to GPU Buffer
        if torch.cuda.is_available():
            model1.cuda()

        param_optimizer = list(model1.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        
        # This variable contains all of the hyperparemeter information our training loop needs
        optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, correct_bias=False)
        #scheduler = WarmupLinearSchedule(optimizer, warmup_steps=100, t_total=1000)
    
        # 1. Model1 BertForSequenceClassification Training
        print ('Start model 1 BertForSequenceClassification Training')
        model1.train()
        for _ in trange(epochs1, desc="Epoch"):    
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels, b_extras = batch
                optimizer.zero_grad()
                outputs = model1(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

                loss, logits = outputs[:2]

                loss.backward()
                optimizer.step()
                #scheduler.step()

                del batch
   
        model1.save_pretrained('./models/len' + str(seq_length) + '/') # save a model

    model1.eval()

    # TwoLayerNet
    N, D_in, H, D_out = batch_size2, input_dim2, hidden_size2, 2
    model2 = TwoLayerNet(D_in, H, D_out)
    optimizer2 = AdamW(model2.parameters(), lr=0.005)
    #scheduler2 = WarmupLinearSchedule(optimizer2, warmup_steps=100, t_total=1000)
    
    if torch.cuda.is_available():
        model2.cuda()

    model2.train()
    # 2. Model2 TwoLayerNet Training
    print ('Start model 2 ThreeLayerNet Training')
    for _ in trange(epochs2, desc="Epoch"):
        tr_loss = 0; nb_tr_steps = 0

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels, b_extras = batch
            optimizer2.zero_grad()
            input = b_extras

            output = model2(input)
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(output, b_labels)
            loss.backward()
            optimizer2.step()
            
            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss/nb_tr_steps))

        # Validation

        # Put model in evaluation mode to evaluate loss on the validation set
        model2.eval()

        # Tracking variables 
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
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

    # Model 3: Ensemble
    N, D_in, H, D_out = batch_size3, input_dim3, hidden_size3, 2
    model3 = TwoLayerNet(D_in, H, D_out)
    optimizer3 = AdamW(model3.parameters(), lr=0.005)

    if torch.cuda.is_available():
        model1.cuda()
        model2.cuda()
        model3.cuda()

    model1.eval()
    model2.eval()
    model3.train()
    # 3. Model3 TwoLayerNet Training
    print ('Start model 3 TwoLayerNet Training')
    for _ in trange(epochs3, desc="Epoch"):
        tr_loss = 0; nb_tr_steps = 0

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels, b_extras = batch

            optimizer3.zero_grad()
            with torch.no_grad():
                outputs = model1(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                loss, logits = outputs[:2]            
                logits = logits.to('cpu').numpy()
                output1 = np.argmax(logits, axis=1)
                output1 = [[float(item)] for item in output1]
                output2 = model2(b_extras)
                output2 = output2.to('cpu').numpy()
                output2 = np.argmax(output2, axis=1)
                output2 = [[float(item)] for item in output2]
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
        output1 = [[float(item)] for item in output1]
        b_extras = b_extras.float()
        output2 = model2(b_extras)
        output2 = output2.to('cpu').numpy()
        output2 = np.argmax(output2, axis=1)
        output2 = [[float(item)] for item in output2]
        output1 = torch.tensor(output1)
        output1 = output1.to(device)
        output2 = torch.tensor(output2)
        output2 = output2.to(device)
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

