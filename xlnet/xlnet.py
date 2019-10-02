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
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score

user_features_fields = ['posts', 'comments']
input_dim = len(user_features_fields)
MAX_LEN = 128
batch_size = 32
epochs = 4 # Number of training epochs (authors recommend between 2 and 4)

if torch.cuda.is_available():
    print ('cuda is available. use gpu.')
    device = torch.device("cuda")
else:
    print ('cuda is not available. use cpu.')
    device = torch.device("cpu")


# START
for seq_length in range(1, 11):
    print ('seq_length: %d'%(seq_length))
    train_set = "data/leaf_depth/seq.learn." + str(seq_length) + ".tsv"
    test_set = "data/leaf_depth/seq.test." + str(seq_length) + ".tsv"

    df = pd.read_csv(train_set, delimiter='\t', header=None, engine='python', names=['sentence_source', 'label', 'label_notes', 'sentence'])
    df = mylib.processDataFrame(df, is_training=True) # Undersampling
    input_ids, attention_masks, labels = mylib.makeBertElements(df, MAX_LEN)

    # Use train_test_split to split our data into train and validation sets for training
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=2018, test_size=0.1)
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=2018, test_size=0.1)

    # Convert all of our data into torch tensors, the required datatype for our model
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    # Select a batch size for training. For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32
    # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
    # with an iterator the entire dataset does not need to be loaded into memory
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    # Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top. 
    model = XLNetForSequenceClassification.from_pretrained("xlnet-base-uncased", num_labels=2)

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
    #scheduler = WarmupLinearSchedule(optimizer, warmup_steps=100, t_total=1000)


    # trange is a tqdm wrapper around the normal python range
    for _ in trange(epochs, desc="Epoch"):
        # Training
      
        # Set our model to training mode (as opposed to evaluation mode)
        model.train()
      
        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss, logits = outputs[:2]

            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            #scheduler.step()

            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss/nb_tr_steps))


        # Validation

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
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                logits = outputs[0]

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_accuracy = mylib.flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

    df = pd.read_csv(test_set, delimiter='\t', header=None, engine='python', names=['sentence_source', 'label', 'label_notes', 'sentence'])
    df = mylib.processDataFrame(test_set, is_training=False)
    input_ids, attention_masks, labels = mylib.makeBertElements(df, MAX_LEN)

    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)
    prediction_labels = torch.tensor(labels)
      
    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)


    # Prediction on test set

    # Put model in evaluation mode
    model.eval()

    # Tracking variables
    predictions , true_labels = [], []

    # Predict
    for batch in prediction_dataloader:
      # Add batch to GPU
      batch = tuple(t.to(device) for t in batch)
      b_input_ids, b_input_mask, b_labels = batch
      
      # Telling the model not to compute or store gradients, saving memory and speeding up prediction
      with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = outputs[0]

      # Move logits and labels to CPU
      logits = logits.detach().cpu().numpy()
      label_ids = b_labels.to('cpu').numpy()
      
      # Store predictions and true labels
      predictions.append(logits)
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

