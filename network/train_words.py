import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Dropout
from keras.callbacks import ModelCheckpoint
from nltk.tokenize import word_tokenize
import gensim

print(Dropout)


def read_file(filename="../sonnets.txt"):
    with open(filename, "r") as corpus_file:
        sorted_words = sorted(
            list(set(word_tokenize(corpus_file.read()))))
        corpus_file.seek(0)
        data = [word_tokenize(k)
                for k in corpus_file.readlines() if len(k.split()) > 1]
        encoding = {c: i for i, c in enumerate(sorted_words)}
        encoding['\n'] = len(sorted_words)
        decoding = {i: c for i, c in enumerate(sorted_words)}
        decoding[len(sorted_words)] = '\n'
        max_sent_len = max([len(sent) for sent in data])
    return data, encoding, decoding, max_sent_len
# Get a unique identifier for each char in the corpus, then make some dicts to ease encoding and decoding
# chars = sorted(list(set(corpus)))
# 300 = len(chars)
# encoding = {c: i for i, c in enumerate(chars)}
# decoding = {i: c for i, c in enumerate(chars)}
# print("Our corpus contains {0} unique characters.".format(300))

# it slices, it dices, it makes julienned datasets!
# chop up our data into X and y, slice into roughly (300 / skip) overlapping 'sentences'
# of length sentence_length, and encode the chars

X_data, encoding, decoding, max_sent_len = read_file()

print(len(encoding))

print('loading word2vec')
model = gensim.models.KeyedVectors.load_word2vec_format(
    '~/GoogleNews-vectors-negative300.bin', binary=True)
print('word2vec loaded !!')


def give_vect(word):
    if word in model:
        return model['word']
    else:
        return np.zeros((300), dtype=np.float32)

sentence_length = max_sent_len
skip = 1
num_sentences = 0
X = []
y = []
print("Vectorizing X and y...")
for i, sentence in enumerate(X_data):
    if not sentence:
        num_sentences -= 1
        continue
    for t, word in enumerate(sentence):
        X_t = np.zeros((sentence_length, 300), dtype=np.float32)
        y_t = np.zeros((len(encoding)), dtype=np.bool)
        X_t[:t + 1] = np.array([give_vect(word) for word in sentence[:t + 1]])
        if t + 1 < len(sentence):
            y_t[encoding[word]] = 1
        else:
            y_t[len(encoding) - 1] = 1
        X.append(X_t)
        y.append(y_t)
        num_sentences += 1


X = np.array(X)
y = np.array(y)
# Define our model
print("Let's build a brain!")
model = Sequential()
model.add(LSTM(len(encoding), input_shape=(sentence_length, 300)))
model.add(Dropout(0.5))
model.add(Dense(len(encoding)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Dump our model architecture to a file so we can load it elsewhere
architecture = model.to_yaml()
with open('model_words.yaml', 'a') as model_file:
    model_file.write(architecture)

# Set up checkpoints
file_path = "weights_words-{epoch:02d}-{loss:.3f}.hdf5"
checkpoint = ModelCheckpoint(
    file_path, monitor="loss", verbose=1, save_best_only=True, mode="min")
callbacks = [checkpoint]

# Action time! [Insert guitar solo here]
model.fit(X, y, nb_epoch=5, batch_size=8, callbacks=callbacks)
