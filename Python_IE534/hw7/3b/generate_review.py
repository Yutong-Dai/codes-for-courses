import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


import logging

log_level = logging.INFO
logger = logging.getLogger()
logger.setLevel(log_level)
handler = logging.FileHandler("3b.log")
handler.setLevel(log_level)
formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

imdb_dictionary = np.load('../preprocessed_data/imdb_dictionary.npy')
vocab_size = 8000 + 1

word_to_id = {token: idx for idx, token in enumerate(imdb_dictionary)}

model = torch.load('language.model')
print('model loaded...')
model.cuda()

model.eval()


def genearate_review(tokens, temperature, model, word_to_id):
    logger.info(tokens[0], tokens[1])
    logger.info(temperature)
    token_ids = np.asarray([[word_to_id.get(token, -1)+1 for token in x] for x in tokens])
    # preload phrase
    x = Variable(torch.LongTensor(token_ids)).cuda()
    embed = model.embedding(x)
    no_of_timesteps = embed.shape[1]
    model.reset_state()
    outputs = []

    for i in range(no_of_timesteps):
        h = model.lstm1(embed[:, i, :])
        h = model.bn_lstm1(h)
        h = model.dropout1(h, dropout=0.3, train=False)
        h = model.lstm2(h)
        h = model.bn_lstm2(h)
        h = model.dropout2(h, dropout=0.3, train=False)
        h = model.lstm3(h)
        h = model.bn_lstm3(h)
        h = model.dropout3(h, dropout=0.3, train=False)
        h = model.decoder(h)
        outputs.append(h)

    outputs = torch.stack(outputs)
    outputs = outputs.permute(1, 2, 0)
    output = outputs[:, :, -1]  # batch_size, vocab_size
    length_of_review = 150
    review = []
    for _ in range(length_of_review):
        # sample a word from the previous output
        output = output/temperature
        probs = torch.exp(output)
        # discard unknown token
        probs[:, 0] = 0.0
        probs = probs/(torch.sum(probs, dim=1).unsqueeze(1))
        x = torch.multinomial(probs, 1)
        review.append(x.cpu().data.numpy()[:, 0])
        # predict the very next word
        embed = model.embedding(x)

        h = model.lstm1(embed[:, 0, :])
        h = model.bn_lstm1(h)
        h = model.dropout1(h, dropout=0.3, train=False)

        h = model.lstm2(h)
        h = model.bn_lstm2(h)
        h = model.dropout2(h, dropout=0.3, train=False)

        h = model.lstm3(h)
        h = model.bn_lstm3(h)
        h = model.dropout3(h, dropout=0.3, train=False)

        output = model.decoder(h)

    review = np.asarray(review)
    review = review.T
    review = np.concatenate((token_ids, review), axis=1)
    review = review - 1
    review[review < 0] = vocab_size - 1
    review_words = imdb_dictionary[review]
    for review in review_words:
        prnt_str = ''
        for word in review:
            prnt_str += word
            prnt_str += ' '
        logger.info(prnt_str)


tokens = [['i', 'love', 'this', 'movie', '.'], ['i', 'hate', 'this', 'movie', '.']]
genearate_review(tokens=tokens, temperature=1.0, model=model, word_to_id=word_to_id)
genearate_review(tokens=tokens, temperature=0.5, model=model, word_to_id=word_to_id)
genearate_review(tokens=tokens, temperature=2, model=model, word_to_id=word_to_id)
