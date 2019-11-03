import pandas as pd
import numpy as np
from pytorch_transformers import *
from keras.preprocessing.sequence import pad_sequences
from sklearn.utils import resample

def processDataFrame(df, is_training):
   df = df.dropna()
   df.label = df.label.astype(int)
   df.label = (df.label + 1) % 2

   df_class1, df_class2 = df[df.label == 0], df[df.label == 1]

   (df_majority, df_minority) = (df_class1, df_class2) if len(df_class1) > len(df_class2) else (df_class2, df_class1)


   if not is_training:
      print ('test datset [%d]: %d, [%d]: %d'%(df_majority.label.values[0], len(df_majority), df_minority.label.values[0], len(df_minority)))
   else:
      print ('train datset [%d]: %d, [%d]: %d'%(df_majority.label.values[0], len(df_majority), df_minority.label.values[0], len(df_minority)))
      #length_minority = 20000 if len(df_minority) > 20000 else len(df_minority)
      length_minority = len(df_minority)

      df_majority_downsampled = resample(df_majority, replace=False, n_samples=length_minority, random_state=123)
      df_minority_downsampled = resample(df_minority, replace=False, n_samples=length_minority, random_state=123)

      df_downsampled = pd.concat([df_majority_downsampled, df_minority_downsampled])
      df = df_downsampled

      print ('train datset [%d]: %d, [%d]: %d'%(df_majority_downsampled.label.values[0], len(df_majority_downsampled), df_minority_downsampled.label.values[0], len(df_minority_downsampled)))

   return df

def processDataFrameVader(df, is_training):
    df = df.dropna()
    
    df.loc[df.label >= 0.05, 'label'] = 2.0
    df.loc[(df.label > -0.05) & (df.label < 0.05), 'label'] = 1.0
    df.loc[df.label <= -0.05, 'label'] = 0.0
    
    df.label = df.label.astype(int)
        
    df_class0 = df[df.label == 0] # negative
    df_class1 = df[df.label == 1] # neutral
    df_class2 = df[df.label == 2] # positive
    
    if len(df_class1) > len(df_class2) and len(df_class0) > len(df_class2):
        df_majority1 = df_class1
        df_majority2 = df_class0
        df_minority = df_class2
    elif len(df_class1) > len(df_class0) and len(df_class2) > len(df_class0):
        df_majority1 = df_class1
        df_majority2 = df_class2
        df_minority = df_class0
    else:
        df_majority1 = df_class0
        df_majority2 = df_class2
        df_minority = df_class1

    if not is_training:
        print ("test dataset [%d]: %d, [%d]: %d, [%d]: %d"%(df_majority1.label.values[0], len(df_majority1), df_majority2.label.values[0], len(df_majority2), df_minority.label.values[0], len(df_minority)))
    else:
        print ("train dataset [%d]: %d, [%d]: %d, [%d]: %d"%(df_majority1.label.values[0], len(df_majority1), df_majority2.label.values[0], len(df_majority2), df_minority.label.values[0], len(df_minority)))

        df_majority1_downsampled = resample(df_majority1,
                                        replace=False,
                                        n_samples=len(df_minority),
                                        random_state=123)
        df_majority2_downsampled = resample(df_majority2,
                                        replace=False,
                                        n_samples=len(df_minority),
                                        random_state=123)
        df_downsampled = pd.concat([df_majority1_downsampled, df_majority2_downsampled, df_minority])

        df = df_downsampled

        print ("train dataset [%d]: %d, [%d]: %d, [%d]: %d"%(df_majority1_downsampled.label.values[0], len(df_majority1_downsampled), df_majority2_downsampled.label.values[0], len(df_majority2_downsampled), df_minority.label.values[0], len(df_minority)))

    return df

def makeBertElements(df, MAX_LEN):
    # Create sentence and label lists
    sentences = df.sentence.values

    # We need to add special tokens at the beginning and end of each sentence for BERT to work properly
    sentences = ["[CLS] " + str(sentence) + " [SEP]" + "NULL" + " [SEP]" for sentence in sentences]
    labels = df.label.values

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    # Set the maximum sequence length. The longest sequence in our training set is 47, but we'll leave room on the end anyway. 
    # In the original paper, the authors used a length of 512.
    # MAX_LEN = 128

    # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
    #input_ids = [tokenizer.create_token_type_ids_from_sequences(sentence, "NULL") for sentence in sentences]
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    type_ids = []
    for item in tokenized_texts:
        type_ids.append([0 for x in item])
        type_ids[-1][-1] = 1
        type_ids[-1][-2] = 1

    type_ids = pad_sequences(type_ids, maxlen=MAX_LEN, dtype="int", truncating="post", padding="post", value=1)

    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

    return input_ids, type_ids, attention_masks, labels

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


