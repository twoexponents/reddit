import pandas as pd
import numpy as np
from pytorch_transformers import *
from keras.preprocessing.sequence import pad_sequences
from sklearn.utils import resample

def processDataFrame(file):
   df = pd.read_csv(file, delimiter='\t', header=None, engine='python', names=['sentence_source', 'label', 'label_notes', 'sentence'])
   df = df.dropna()
   df.label = df.label.astype(int)

   df_class1, df_class2 = df[df.label == 0], df[df.label == 1]

   (df_majority, df_minority) = (df_class1, df_class2) if len(df_class1) > len(df_class2) else (df_class2, df_class1)

   print ('train datset [%d]: %d, [%d]: %d'%(df_majority.label.values[0], len(df_majority), df_minority.label.values[0], len(df_minority)))

   if 'learn' in file:
      length_minority = 20000 if len(df_minority) > 20000 else len(df_minority)

      df_majority_downsampled = resample(df_majority, replace=False, n_samples=length_minority, random_state=123)
      df_minority_downsampled = resample(df_minority, replace=False, n_samples=length_minority, random_state=123)

      df_downsampled = pd.concat([df_majority_downsampled, df_minority_downsampled])
      df = df_downsampled

      print ('train datset [%d]: %d, [%d]: %d'%(df_majority_downsampled.label.values[0], len(df_majority_downsampled), df_minority_downsampled.label.values[0], len(df_minority_downsampled)))

   return df

def makeBertElements(df, MAX_LEN):
    # Create sentence and label lists
    sentences = df.sentence.values

    # We need to add special tokens at the beginning and end of each sentence for BERT to work properly
    sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in sentences]
    labels = df.label.values

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    # Set the maximum sequence length. The longest sequence in our training set is 47, but we'll leave room on the end anyway. 
    # In the original paper, the authors used a length of 512.
    # MAX_LEN = 128

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

    return input_ids, attention_masks, labels

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


