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
input_dim = len(user_features_fields)
MAX_LEN = 128
batch_size = 32
epochs = 4

with open('/home/jhlim/SequencePrediction/data/userfeatures.activity.p', 'rb') as f:
    d_userfeatures = pickle.load(f)

if torch.cuda.is_available():
    print ('cuda is available. use gpu.')
    device = torch.device("cuda")
else:
    print ('cuda is not available. use cpu.')
    device = torch.device("cpu")


# START
for seq_length in range(1, 3):
    print ('seq_length: %d'%(seq_length))
    train_set = "data/leaf_depth/seq.learn." + str(seq_length) + ".tsv"
    test_set = "data/leaf_depth/seq.test." + str(seq_length) + ".tsv"

    df = pd.read_csv(train_set, delimiter='\t', header=None, engine='python', names=['sentence_source', 'label', 'label_notes', 'sentence'])
    df = mylib.processDataFrame(df, is_training=True) # Undersampling
    input_ids, attention_masks, labels = mylib.makeBertElements(df, MAX_LEN)

    # jhlim: additional features dataset
    element_list = df.sentence_source.values
    extra_features = []
    for element in element_list:
        user_features = [0.0]*len(user_features_fields)
        if element in d_userfeatures:
            user_features = d_userfeatures[element]['user'][0:2]

        extra_features.append(user_features)

    # Use train_test_split to split our data into train and validation sets for training
    train_inputs, validation_inputs, train_labels, validation_labels, train_extras, validation_extras = train_test_split(input_ids, labels, extra_features, random_state=2018, test_size=0.1)
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                 random_state=2018, test_size=0.1)

    # Convert all of our data into torch tensors, the required datatype for our model
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)
    train_extras = torch.tensor(train_extras)
    validation_extras = torch.tensor(validation_extras)

    # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
    # with an iterator the entire dataset does not need to be loaded into memory
    train_data = TensorDataset(train_inputs, train_masks, train_labels, train_extras)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels, validation_extras)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    # Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top. 
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    # Load model parameters to GPU Buffer
    if torch.cuda.is_available():
        model.cuda()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    
    # This variable contains all of the hyperparemeter information our training loop needs
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, correct_bias=False)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=100, t_total=1000)
    
    # TwoLayerNet
    N, D_in, H, D_out = batch_size, 2, 100, 2
    model2 = TwoLayerNet(D_in, H, D_out)
    optimizer2 = AdamW(model2.parameters(), lr=0.005)
    scheduler2 = WarmupLinearSchedule(optimizer2, warmup_steps=100, t_total=1000)

    
    # 1. Model1 BertForSequenceClassification Training
    print ('Start model 1 BertForSequenceClassification Training')
    for _ in trange(epochs, desc="Epoch"):
        model.train()
        
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels, b_extras = batch
            optimizer.zero_grad()
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

            loss, logits = outputs[:2]

            loss.backward()
            optimizer.step()
            scheduler.step()
    
    model.eval()
    model = model.to('cpu')

    if torch.cuda.is_available():
        model2.cuda()

    model2.train()
    # 2. Model2 TwoLayerNet Training
    print ('Start model 2 TwoLayerNet Training')
    for _ in trange(epochs, desc="Epoch"):
        try:
            tr_loss = 0; nb_tr_steps = 0

            # Train the data for one epoch
            for step, batch in enumerate(train_dataloader):
                #batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels, b_extras = batch
                # Clear out the gradients (by default they accumulate)
                optimizer2.zero_grad()
                with torch.no_grad():
                    outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                loss, logits = outputs[:2]
                logits = logits.to(device)
                b_labels = b_labels.to(device)
                
                #loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                predicts = model2(logits)
                #last_hidden_states = outputs[0]
                #mean_hidden_states = torch.mean(last_hidden_states, 1, keepdim=False)
                #input = torch.cat((mean_hidden_states, b_extras), dim=1)
                #input = b_extras
                #input = mean_hidden_states

                #predicts = model2(input)
                loss_fn = torch.nn.CrossEntropyLoss()
                loss2 = loss_fn(predicts, b_labels)
                loss2.backward()
                optimizer2.step()
                scheduler2.step()

                #mean_hidden_states = mean_hidden_states.detach().cpu().numpy()

                # Update tracking variables
                tr_loss += loss2.item()
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
                #batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels, b_extras = batch
                with torch.no_grad():
                    outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                    loss, logits = outputs[:2]
                    logits = logits.to(device)
                    b_labels = b_labels.to(device) 

                    # Forward pass, calculate logit predictions
                    predicts = model2(logits)
                    #last_hidden_states = logits[0]
                    #mean_hidden_states = torch.mean(last_hidden_states, 1, keepdim=False)
                    #input = torch.cat((mean_hidden_states, b_extras), dim=1)
                    #input = b_extras
                    #input = mean_hidden_states
                    
                    #predicts = model2(input)
                    
                # Move logits and labels to CPU
                predicts = predicts.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                tmp_eval_accuracy = mylib.flat_accuracy(predicts, label_ids)

                eval_accuracy += tmp_eval_accuracy
                nb_eval_steps += 1

            print ("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

        except Exception as e:
            print (e)
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            sys.exit()

    df = pd.read_csv(test_set, delimiter='\t', header=None, engine='python', names=['sentence_source', 'label', 'label_notes', 'sentence'])
    df = mylib.processDataFrame(df, is_training=False)
    input_ids, attention_masks, labels = mylib.makeBertElements(df, MAX_LEN)

    element_list = df.sentence_source.values
    extra_features = []
    for element in element_list:
        user_features = [0.0]*len(user_features_fields)
        if element in d_userfeatures:
            user_features = d_userfeatures[element]['user'][0:2]

        extra_features.append(user_features)

    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)
    prediction_labels = torch.tensor(labels)
    prediction_extras = torch.tensor(extra_features)
      
    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels, prediction_extras)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    # Prediction on test set

    # Put model in evaluation mode
    model2.eval()

    # Tracking variables
    predictions , true_labels = [], []

    # Predict
    for batch in prediction_dataloader:
      b_input_ids, b_input_mask, b_labels, b_extras = batch
      # Telling the model not to compute or store gradients, saving memory and speeding up prediction
      with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss, logits = outputs[:2]
        logits = logits.to(device)

        # Forward pass, calculate logit predictions
        predicts = model2(logits)
        #last_hidden_states = logits[0]
        #mean_hidden_states = torch.mean(last_hidden_states, 1, keepdim=False)
        #input = b_extras
        #input = torch.cat((mean_hidden_states, b_extras), dim=1)
        #input = mean_hidden_states

        #predicts = model2(input)

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


