
import tensorflow as tf
import torch
from pytorch_transformers import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
#from pytorch_pretrained_bert import BertTokenizer, BertConfig
#from pytorch_pretrained_bert import BertAdam#, BertForSequenceClassification
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle

from twoLayerNet import TwoLayerNet

from sklearn.utils import resample
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
#% matplotlib inline

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
for seq_length in range(1, 2):
    print ('seq_length: %d'%(seq_length))
    train_set = "data/leaf_depth/seq.learn." + str(seq_length) + ".tsv"
    test_set = "data/leaf_depth/seq.test." + str(seq_length) + ".tsv"

    df = pd.read_csv(train_set, delimiter='\t', header=None, engine='python', names=['sentence_source', 'label', 'label_notes', 'sentence'])

    #df = df.fillna(0)
    df.dropna()
    df.label = df.label.astype(int)

    print ('shape:', df.shape)
    #print (df.sample(10))

    # jhlim: undersampling training set using dataframe
    df_class1 = df[df.label == 0]
    df_class2 = df[df.label == 1]

    if len(df_class1) > len(df_class2):
        df_majority = df_class1
        df_minority = df_class2
    else:
        df_majority = df_class2
        df_minority = df_class1

    print ("train dataset [%d]: %d, [%d]: %d"%(df_majority.label.values[0], len(df_majority), df_minority.label.values[0], len(df_minority)))

    length_minority = 30000 if len(df_minority) > 30000 else len(df_minority)
    
    df_majority_downsampled = resample(df_majority, replace=False, n_samples=length_minority, random_state=123)
    df_minority_downsampled = resample(df_minority, replace=False, n_samples=length_minority, random_state=123)
    df_downsampled = pd.concat([df_majority_downsampled, df_minority_downsampled])
    df = df_downsampled

    print ("train dataset [%d]: %d, [%d]: %d"%(df_majority_downsampled.label.values[0], len(df_majority_downsampled), df_minority_downsampled.label.values[0], len(df_minority_downsampled)))


    # Create sentence and label lists
    sentences = df.sentence.values

    # We need to add special tokens at the beginning and end of each sentence for BERT to work properly
    sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in sentences]
    labels = df.label.values

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    #print ("Tokenize the first sentence:")
    #print (tokenized_texts[0])


    # Set the maximum sequence length. The longest sequence in our training set is 47, but we'll leave room on the end anyway. 
    # In the original paper, the authors used a length of 512.
    MAX_LEN = 128

    # Pad our input tokens
    #input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
    #                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

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

    # Select a batch size for training. For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32
    batch_size = 32

    # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
    # with an iterator the entire dataset does not need to be loaded into memory

    train_data = TensorDataset(train_inputs, train_masks, train_labels, train_extras)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels, validation_extras)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    # Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top. 
    #model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model = BertModel.from_pretrained("bert-base-uncased")

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
    optimizer = AdamW(optimizer_grouped_parameters,
                         lr=2e-5,
                         correct_bias=False)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=100, t_total=1000)

    # Store our loss and accuracy for plotting
    train_loss_set = []

    # Number of training epochs (authors recommend between 2 and 4)
    epochs = 4

    # trange is a tqdm wrapper around the normal python range
    for _ in trange(epochs, desc="Epoch"):
        # Training
      
        # Set our model to training mode (as opposed to evaluation mode)
        #model.train()
      
        # Tracking variables
        tr_loss = 0
        nb_tr_steps = 0

        # TwoLayerNet
        N, D_in, H, D_out = 32, 770, 50, 2
        model2 = TwoLayerNet(D_in, H, D_out)
        if torch.cuda.is_available():
            model2.cuda()

        model2.train()

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels, b_extras = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            #loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            #train_loss_set.append(loss.item()) 
            outputs = model(b_input_ids)
            last_hidden_states = outputs[0]
            mean_hidden_states = torch.mean(last_hidden_states, 1, keepdim=False)
            input = torch.cat((mean_hidden_states, b_extras), dim=1)

            predicts = model2(input)
            loss = torch.nn.CrossEntropyLoss()
            output = loss(predicts, b_labels)
            output.backward()
            optimizer.step()
            scheduler.step()

            #mean_hidden_states = mean_hidden_states.detach().cpu().numpy()

            # Update tracking variables
            tr_loss += output.item()
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss/nb_tr_steps))

        # Validation

        # Put model in evaluation mode to evaluate loss on the validation set
        #model.eval()

        # Tracking variables 
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels, b_extras = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                logits = model(b_input_ids)
                last_hidden_states = logits[0]
                mean_hidden_states = torch.mean(last_hidden_states, 1, keepdim=False)
                input = torch.cat((mean_hidden_states, b_extras), dim=1)
                
                predicts = model2(input)

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
    df_class1 = df[df.label == 0]
    df_class2 = df[df.label == 1]

    print ("test dataset [%d]: %d, [%d]: %d"%(df_class1.label.values[0], len(df_class1), df_class2.label.values[0], len(df_class2)))
    # Create sentence and label lists
    sentences = df.sentence.values

    # We need to add special tokens at the beginning and end of each sentence for BERT to work properly
    sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in sentences]
    labels = df.label.values

    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]


    MAX_LEN = 128
    # Pad our input tokens
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
      seq_mask = [float(i>0) for i in seq]
      attention_masks.append(seq_mask) 

    # jhlim: additional features dataset
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
      
    batch_size = 32  

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
      # Add batch to GPU
      batch = tuple(t.to(device) for t in batch)
      # Unpack the inputs from our dataloader
      b_input_ids, b_input_mask, b_labels, b_extras = batch
      # Telling the model not to compute or store gradients, saving memory and speeding up prediction
      with torch.no_grad():
        # Forward pass, calculate logit predictions
        logits = model(b_input_ids)
        last_hidden_states = logits[0]
        mean_hidden_states = torch.mean(last_hidden_states, 1, keepdim=False)
        input = torch.cat((mean_hidden_states, b_extras), dim=1)

        predicts = model2(input)

      # Move logits and labels to CPU
      predicts = predicts.detach().cpu().numpy()
      label_ids = b_labels.to('cpu').numpy()
      
      # Store predictions and true labels
      predictions.append(predicts)
      true_labels.append(label_ids)

    '''
    from sklearn.metrics import matthews_corrcoef
    matthews_set = []

    for i in range(len(true_labels)):
        matthews = matthews_corrcoef(true_labels[i],
                    np.argmax(predictions[i], axis=1).flatten())
        matthews_set.append(matthews)

    print (matthews_corrcoef(flat_true_labels, flat_predictions))
    '''

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


