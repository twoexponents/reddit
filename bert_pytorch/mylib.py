import pandas as pd
import numpy as np
from pytorch_transformers import *
from keras.preprocessing.sequence import pad_sequences
from sklearn.utils import resample

class MyBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(MyBertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
            position_ids=None, head_mask=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)
        
        pooled_output = outputs[1]

        return pooled_output
        '''
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]

        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs # (loss), logits, (hidden_states), (attentions)
        '''

def processDataFrame(df, is_training):
   df = df.dropna()
   df.label = df.label.astype(int)

   df_class1, df_class2 = df[df.label == 0], df[df.label == 1]

   (df_majority, df_minority) = (df_class1, df_class2) if len(df_class1) > len(df_class2) else (df_class2, df_class1)


   if not is_training:
      print ('test dataset [%d]: %d, [%d]: %d'%(df_majority.label.values[0], len(df_majority), df_minority.label.values[0], len(df_minority)))
   else:
      print ('train dataset [%d]: %d, [%d]: %d'%(df_majority.label.values[0], len(df_majority), df_minority.label.values[0], len(df_minority)))
      length_minority = 20000 if len(df_minority) > 20000 else len(df_minority)
      #length_minority = len(df_minority)

      df_majority_downsampled = resample(df_majority, replace=False, n_samples=length_minority, random_state=123)
      df_minority_downsampled = resample(df_minority, replace=False, n_samples=length_minority, random_state=123)

      df_downsampled = pd.concat([df_majority_downsampled, df_minority_downsampled])
      df = df_downsampled

      print ('train dataset [%d]: %d, [%d]: %d'%(df_majority_downsampled.label.values[0], len(df_majority_downsampled), df_minority_downsampled.label.values[0], len(df_minority_downsampled)))

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

def makeBertElement(sentence, tokenizer, MAX_LEN):
    sentence = "[CLS] " + sentence + " [SEP]"
    tokenized_texts = tokenizer.tokenize(sentence)
    input_id = tokenizer.convert_tokens_to_ids(tokenized_texts)
    input_ids = pad_sequences([input_id], maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

    return input_ids, attention_masks

def makeBertElementsList(df, MAX_LEN):

    input_ids = []
    attention_masks = []
    learn_instances = df.sentence.values
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    for sentences in learn_instances:
        sub_x = []
        sub_attention = []
        for sentence in sentences:
            temp = "[CLS]" + str(sentence) + " [SEP]"
            tokenized_text = tokenizer.tokenize(temp)

            input_id = tokenizer.convert_tokens_to_ids(tokenized_text)

            sub_x.append(input_id)

            temp_mask = [float(i>0) for i in input_id]
            sub_attention.append(temp_mask)
            
        sub_x = pad_sequences(sub_x, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
        input_ids.append(sub_x)
        attention_masks.append(sub_attention)

    labels = df.label.values
    return input_ids, attention_masks, labels

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


