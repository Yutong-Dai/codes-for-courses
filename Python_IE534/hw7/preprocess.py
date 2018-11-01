'''
File: preprocess.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: Sunday, 2018-10-28 20:48
Last Modified: Monday, 2018-10-29 15:31
--------------------------------------------
Desscription: Preprocess the Imbd review.
'''

import numpy as np
import os
import nltk
import itertools
import io

if(not os.path.isdir('preprocessed_data')):
    os.mkdir('preprocessed_data')

# get all of the training reviews (including unlabeled reviews)
train_directory = '/projects/training/bauh/NLP/aclImdb/train/'
pos_filenames = os.listdir(train_directory + 'pos/')
neg_filenames = os.listdir(train_directory + 'neg/')
unsup_filenames = os.listdir(train_directory + 'unsup/')

pos_filenames = [train_directory+'pos/'+filename for filename in pos_filenames]  # 12500
neg_filenames = [train_directory+'neg/'+filename for filename in neg_filenames]  # 12500
unsup_filenames = [train_directory+'unsup/'+filename for filename in unsup_filenames]  # 50000

filenames = pos_filenames + neg_filenames + unsup_filenames

print("working on the traning data...")
counter = 0
x_train = []
for filename in filenames:
    with io.open(filename, 'r', encoding='utf-8') as f:
        line = f.readlines()[0]
    line = line.replace('<br />', ' ')
    line = line.replace('\x96', ' ')
    # tokenize each review
    line = nltk.word_tokenize(line)
    line = [w.lower() for w in line]
    x_train.append(line)
    counter += 1
    if (counter % 1000 == 0):
        print("process at the {} review.".format(counter))

# word_to_id and id_to_word. associate an id to every unique token in the training data
all_tokens = itertools.chain.from_iterable(x_train)
word_to_id = {token: idx for idx, token in enumerate(set(all_tokens))}

# must recall itertools.chain.from_iterable since all_tokens will be empty after above operation
all_tokens = itertools.chain.from_iterable(x_train)
id_to_word = [token for idx, token in enumerate(set(all_tokens))]
id_to_word = np.asarray(id_to_word)

# let's sort the indices by word frequency instead of random
x_train_token_ids = [[word_to_id[token] for token in x] for x in x_train]
count = np.zeros(id_to_word.shape)
for x in x_train_token_ids:
    for token in x:
        count[token] += 1
indices = np.argsort(-count)
id_to_word = id_to_word[indices]
count = count[indices]


# recreate word_to_id based on sorted list
word_to_id = {token: idx for idx, token in enumerate(id_to_word)}

# reserve id=0 for an unknown token, achiving this by shifting id by 1
# Just be aware of this that `id_to_word` is now off by 1 index if you actually want to convert ids to words.
x_train_token_ids = [[word_to_id.get(token, -1)+1 for token in x] for x in x_train]


# save dictionary
np.save('preprocessed_data/imdb_dictionary.npy', np.asarray(id_to_word))
print("dictionary saved!")
# save training data to single text file
with io.open('preprocessed_data/imdb_train.txt', 'w', encoding='utf-8') as f:
    for tokens in x_train_token_ids:
        for token in tokens:
            f.write("%i " % token)
        f.write("\n")
print("imdb_train.txt saved!")

print("working on the testing data...")
# get all of the test reviews
test_directory = '/projects/training/bauh/NLP/aclImdb/test/'
pos_filenames = os.listdir(test_directory + 'pos/')
neg_filenames = os.listdir(test_directory + 'neg/')
pos_filenames = [test_directory+'pos/'+filename for filename in pos_filenames]
neg_filenames = [test_directory+'neg/'+filename for filename in neg_filenames]
filenames = pos_filenames+neg_filenames

counter = 0
x_test = []
for filename in filenames:
    with io.open(filename, 'r', encoding='utf-8') as f:
        line = f.readlines()[0]
    line = line.replace('<br />', ' ')
    line = line.replace('\x96', ' ')
    line = nltk.word_tokenize(line)
    line = [w.lower() for w in line]
    x_test.append(line)
    counter += 1
    if (counter % 1000 == 0):
        print("process at the {} review.".format(counter))

x_test_token_ids = [[word_to_id.get(token, -1)+1 for token in x] for x in x_test]

# save test data to single text file
with io.open('preprocessed_data/imdb_test.txt', 'w', encoding='utf-8') as f:
    for tokens in x_test_token_ids:
        for token in tokens:
            f.write("%i " % token)
        f.write("\n")
print("imdb_test.txt saved!")


print("Working on glove!")
glove_filename = '/projects/training/bauh/NLP/glove.840B.300d.txt'
with io.open(glove_filename, 'r', encoding='utf-8') as f:
    lines = f.readlines()

glove_dictionary = []
glove_embeddings = []
count = 0
for line in lines:
    line = line.strip()
    line = line.split(' ')
    glove_dictionary.append(line[0])
    embedding = np.asarray(line[1:], dtype=np.float)
    glove_embeddings.append(embedding)
    count += 1
    if(count >= 100000):
        break

glove_dictionary = np.asarray(glove_dictionary)
glove_embeddings = np.asarray(glove_embeddings)
# added a vector of zeros for the unknown tokens
glove_embeddings = np.concatenate((np.zeros((1, 300)), glove_embeddings))

word_to_id = {token: idx for idx, token in enumerate(glove_dictionary)}

x_train_token_ids = [[word_to_id.get(token, -1)+1 for token in x] for x in x_train]
x_test_token_ids = [[word_to_id.get(token, -1)+1 for token in x] for x in x_test]
np.save('preprocessed_data/glove_dictionary.npy', glove_dictionary)
print("glove_dictionary saved!")
np.save('preprocessed_data/glove_embeddings.npy', glove_embeddings)
print("glove_embeddings saved!")

with io.open('preprocessed_data/imdb_train_glove.txt', 'w', encoding='utf-8') as f:
    for tokens in x_train_token_ids:
        for token in tokens:
            f.write("%i " % token)
        f.write("\n")

with io.open('preprocessed_data/imdb_test_glove.txt', 'w', encoding='utf-8') as f:
    for tokens in x_test_token_ids:
        for token in tokens:
            f.write("%i " % token)
        f.write("\n")