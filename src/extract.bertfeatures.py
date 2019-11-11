import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import sys
import math
from pytorch_transformers import *
from keras.preprocessing.sequence import pad_sequences
from sklearn.utils import resample
from tqdm import tqdm, trange
from myloaddatalib import load_userfeatures, load_contfeatures, load_timefeatures

MAX_LEN = 128
seq_length = 2 
batch_size = 400

def main():
    with open('/home/jhlim/data/commentbodyfeatures.p', 'rb') as f:
        sentencefile = pickle.load(f)

    f = open('/home/jhlim/data/seq.learn.%d.csv'%(seq_length), 'r')
    learn_instances = list(map(lambda x:x.replace('\n', '').split(','), f.readlines()))
    f.close()
    f = open('/home/jhlim/data/seq.test.%d.csv'%(seq_length), 'r')
    test_instances = list(map(lambda x:x.replace('\n', '').split(','), f.readlines()))
    f.close()

    d_user = load_userfeatures()
    d_liwc = load_contfeatures()
    d_time = load_timefeatures()

    tuples = []
    for seq in learn_instances:
        for i, element in enumerate(seq):
            if False in list(map(lambda x:element in x, [d_user, d_liwc, d_time])):
                continue
            if i > (seq_length-1):
                break
            if element in sentencefile:
                tuples.append((element, sentencefile[element]))
    for seq in test_instances:
        for i, element in enumerate(seq):
            if False in list(map(lambda x:element in x, [d_user, d_liwc, d_time])):
                continue
            if i > (seq_length-1):
                break
            if element in sentencefile:
                tuples.append((element, sentencefile[element]))

    print ('making input files done.')

    model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
    
    if torch.cuda.is_available():
        device = 'cuda'
        model.cuda()
    else:
        device = 'cpu'

    d_bertfeatures = {}

    elementSentence = {}
    elements = []; sentences = []
    for tuple in tuples:
        element, sentence = tuple
        if element not in elementSentence:
            elementSentence[element] = "[CLS] " + str(sentence) + " [SEP]"

    elements = list(elementSentence.keys())
    sentences = list(elementSentence.values())

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    attention_masks = []

    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

    input_ids = torch.tensor(input_ids).to(device)
    attention_masks = torch.tensor(attention_masks).to(device)

    model.eval()

    flag = True
    print (len(elements))

    
    for i in trange(math.ceil(len(input_ids)/batch_size), desc="batch"):
        batch_elements = elements[i*batch_size:(i+1)*batch_size]
        inputs = input_ids[i*batch_size:(i+1)*batch_size]
        attentions = attention_masks[i*batch_size:(i+1)*batch_size]
        with torch.no_grad():
            outputs = model(inputs, token_type_ids=None, attention_mask=attentions)
            if flag:
                print (outputs[0].shape)
            outputs = outputs[0]
            mean_outputs = torch.mean(outputs, 1, keepdim=False)
            if flag:
                flag = False
                print (mean_outputs.shape)
            mean_outputs = mean_outputs.to('cpu').tolist()

        if len(batch_elements) != len(mean_outputs):
            print ('the size is different. elements: %d, outputs: %d.'%(len(elements), len(mean_outputs)))
            sys.exit(-1)

        for element, feature in zip(batch_elements, mean_outputs):
            d_bertfeatures[element] = feature


    print ('size of d_bertfeatures: ', len(d_bertfeatures))

    pickle.dump(d_bertfeatures, open('/home/jhlim/data/bertfeatures' + str(seq_length) + '.p', 'wb'))


if __name__ == "__main__":
    main()
