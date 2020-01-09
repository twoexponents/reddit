import torch
from pytorch_transformers import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
import pandas as pd
import numpy as np
import pickle
import mylib
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
from sklearn.metrics import classification_report

MAX_LEN = 128 # 128
batch_size = 32 # For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32
epochs = 4 # Number of training epochs (authors recommend between 2 and 4)
TRAIN_SIZE = 100000
TEST_SIZE = 20000

torch.manual_seed(123)
torch.cuda.manual_seed(123)

if torch.cuda.is_available():
    print ('cuda is available. use gpu.')
    device = torch.device("cuda")
else:
    print ('cuda is not available. use cpu.')
    device = torch.device("cpu")


# START
for seq_length in range(10, 11):
    print ('seq_length: %d'%(seq_length))
    train_set = "data/dynamic/seq.learn.less" + str(seq_length) + ".tsv"
    test_set = "data/dynamic/seq.test.less" + str(seq_length) + ".tsv"

    train_set = train_set[:TRAIN_SIZE]
    test_set = test_set[:TEST_SIZE]

    df = pd.read_csv(train_set, delimiter='\t', header=None, engine='python', names=['sentence_source', 'label', 'label_notes', 'sentence'])
    df = mylib.processDataFrame(df, is_training=True) # Undersampling
    input_ids, attention_masks, labels = mylib.makeBertElements(df, MAX_LEN)

    train_inputs = input_ids; train_labels = labels
    train_masks = attention_masks

    train_inputs = torch.tensor(train_inputs)
    train_labels = torch.tensor(train_labels)
    train_masks = torch.tensor(train_masks)

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    del df, train_inputs, train_masks, train_labels, train_set
    # Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top. 
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2, output_hidden_states=True)
    
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
      
        model.train()
      
        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            optimizer.zero_grad()
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss, logits = outputs[:2]

            loss.backward()
            optimizer.step()
            #scheduler.step()

            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss/nb_tr_steps))

    df = pd.read_csv(test_set, delimiter='\t', header=None, engine='python', names=['sentence_source', 'label', 'label_notes', 'sentence'])
    df = mylib.processDataFrame(df, is_training=False)
    input_ids, attention_masks, labels = mylib.makeBertElements(df, MAX_LEN)

    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)
    prediction_labels = torch.tensor(labels)
      
    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)


    del prediction_inputs, prediction_masks, prediction_labels, test_set

    # Prediction on test set

    # Put model in evaluation mode
    model.eval()

    # Tracking variables
    predictions , true_labels = [], []

    # Predict
    for batch in prediction_dataloader:
      batch = tuple(t.to(device) for t in batch)
      b_input_ids, b_input_mask, b_labels = batch
      
      # Telling the model not to compute or store gradients, saving memory and speeding up prediction
      with torch.no_grad():
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


    predicts = []; i = 0
    for v1, v2 in zip(flat_true_labels, flat_predictions):
        if v1 == v2:
            decision = True
        else:
            decision = False
            #print ('predicts: ', predictions[i], ', label: ', flat_true_labels[i])
        #decision = True if v1 == v2 else False
        predicts.append(decision)
        i += 1

    ftl = flat_true_labels
    fpd = flat_predictions

    print ('# predicts: %d, # corrects: %d, # 0: %d, # 1: %d, acc: %f, auc: %f'%
        (len(predicts), len(list(filter(lambda x:x, predicts))), len(list(filter(lambda x:x == 0, flat_predictions))), len(list(filter(lambda x:x == 1, fpd))), accuracy_score(ftl, fpd), roc_auc_score(ftl, fpd)))
    print (classification_report(ftl, fpd, labels=[0, 1]))

    model.save_pretrained('./models/len' + str(seq_length) + '/') # Save the model


