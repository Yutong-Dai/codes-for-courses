import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.distributed as dist

import h5py
import time
import os
import io

import sys

imdb_dictionary = np.load('../preprocessed_data/imdb_dictionary.npy')
vocab_size = 8000 + 1

word_to_id = {token: idx for idx, token in enumerate(imdb_dictionary)}

model = torch.load('language.model')
print('model loaded...')
model.cuda()

model.eval()

## create partial sentences to "prime" the model
## this implementation requires the partial sentences
## to be the same length if doing more than one
# tokens = [['i','love','this','movie','.'],['i','hate','this','movie','.']]
tokens = [['a'],['i']]

token_ids = np.asarray([[word_to_id.get(token,-1)+1 for token in x] for x in tokens]) #2, 1

## preload phrase
x = Variable(torch.LongTensor(token_ids)).cuda() #2, 1

embed = model.embedding(x) # batch_size, time_steps, features; 2, 1, 500

state_size = [embed.shape[0],embed.shape[2]] # batch_size, features
no_of_timesteps = embed.shape[1] #1

model.reset_state()

tokens = [['a'],['i']]
token_ids = np.asarray([[word_to_id.get(token,-1)+1 for token in x] for x in tokens]) #2, 1
## preload phrase
x = Variable(torch.LongTensor(token_ids)).cuda() #2, 1
embed = model.embedding(x) # batch_size, time_steps, features; 2, 1, 500
outputs = []
for i in range(no_of_timesteps):
    h = model.lstm1(embed[:,i,:]) #input, batch_size*features, 2*500, output, 2*500
    h = model.bn_lstm1(h) # 2*500
    h = model.dropout1(h,dropout=0.3,train=False) # 2*500
    h = model.lstm2(h) # 2*500
    h = model.bn_lstm2(h) # 2*500
    h = model.dropout2(h,dropout=0.3,train=False) # 2*500
    h = model.lstm3(h) # 2*500
    h = model.bn_lstm3(h) # 2*500
    h = model.dropout3(h,dropout=0.3,train=False) # 2*500
    h = model.decoder(h) # 2*8001
    outputs.append(h)

outputs = torch.stack(outputs) #1, 2, 8001
outputs = outputs.permute(1,2,0) # 2, 8001, 1
output = outputs[:,:,-1] #batch_size, vocab_size, 2, 8001

temperature = 1.0 # float(sys.argv[1])
length_of_review = 150

review = []
####
for j in range(length_of_review):

    ## sample a word from the previous output
    output = output/temperature #2,8001
    probs = torch.exp(output) #2, 8001
    probs[:,0] = 0.0 #2, 8001, make sue we don't choose the unknown token which represented by 0
    probs = probs/(torch.sum(probs,dim=1).unsqueeze(1)) # dim = 1 means by row
    x = torch.multinomial(probs,1) #2, 1
    review.append(x.cpu().data.numpy()[:,0]) #2

    ## predict the next word
    embed = model.embedding(x) #2, 1, 500

    h = model.lstm1(embed[:, 0, :])
    h = model.bn_lstm1(h)
    h = model.dropout1(h,dropout=0.3,train=False)

    h = model.lstm2(h)
    h = model.bn_lstm2(h)
    h = model.dropout2(h,dropout=0.3,train=False)

    h = model.lstm3(h)
    h = model.bn_lstm3(h)
    h = model.dropout3(h,dropout=0.3,train=False)

    output = model.decoder(h)

review = np.asarray(review)
review = review.T
review = np.concatenate((token_ids,review),axis=1)
review = review - 1
review[review<0] = vocab_size - 1
review_words = imdb_dictionary[review]
for review in review_words:
    prnt_str = ''
    for word in review:
        prnt_str += word
        prnt_str += ' '
    print(prnt_str)