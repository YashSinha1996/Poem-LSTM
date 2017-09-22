import numpy as np
from keras.models import load_model
from random import randint, random as rd
from keras.callbacks import ModelCheckpoint
from nltk.tokenize import word_tokenize
import gensim


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


class GenerativeNetwork:

    def give_vect(self, word):
        if word in self.w2v:
            return self.w2v[word]
        else:
            return np.zeros((300), dtype=np.float32)

    def __init__(self, corpus_path, model_path="model_words.yaml", weights_path="weights_words.hdf5"):
        print("ingenerative")
        print('loading word2vec')
        self.w2v = gensim.models.KeyedVectors.load_word2vec_format(
            '/home/yash/Downloads/Work/GoogleNews-vectors-negative300.bin', binary=True)
        print('word2vec loaded !!')

        # Get a unique identifier for each char in the corpus,
        # then make some dicts to ease encoding and decoding
        self.chars, self.encoding, self.decoding, self.sentence_length = read_file()

        # Some fields we'll need later
        self.num_chars = len(self.decoding)
        self.sentence_length_max = 6
        # self.corpus_length = len(self.corpus)

        # Build our network from loaded architecture and weights
        with open(model_path) as model_file:
            architecture = model_file.read()

        # self.model = model_from_yaml(architecture)
        # self.model.load_weights(weights_path)
        self.model=load_model(weights_path)
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def generate(self, seed_pattern=None, error=0):

        X, seed_len = self.make_seed(seed_pattern, error=error)
        generated_text = []
        generated_line = ""
        min_len = 4
        for i in range(randint(5*6, 15*6)):
            # print(X.shape)
            predited = self.model.predict(X, verbose=0)
            # print(X, predited)
            prediction_index = int("".join([str(int(round(a))) for a in predited[0]]))
            print(predited,prediction_index,prediction_index % len(self.decoding))
            prediction_index = prediction_index % len(self.decoding)
            # print(prediction_index)
            # if len(generated_line.split()) < min_len:
            #     prediction_index = np.argmax(predited[0][:-1]) + 1
            # print(prediction_index)
            # generated_line += self.decoding[prediction_index]
            if self.decoding[prediction_index] == '\n' or len(generated_line.split()) > self.sentence_length_max:
                generated_line += "\n"
                words = generated_line.split()
                no_prev_line = len(words)
                if(no_prev_line > 3):
                    # X = np.zeros((1, self.sentence_length, 300),
                    #              dtype=np.float32)
                    X=[]
                    X.append( self.give_vect(words[0]))
                    X.append( self.give_vect(words[1]))
                    X.append( self.give_vect(words[-2]))
                    X.append( self.give_vect(words[-1]))
                    X=np.array([X])
                    seed_len = 4
                elif no_prev_line > 0:
                    # X = np.zeros((1, self.sentence_length, 300),
                    #              dtype=np.float32)
                    X = []
                    for j, word in enumerate(words):
                        X.append(self.give_vect(word))
                    X = np.array([X])
                    seed_len = no_prev_line
                else:
                    X, seed_len = self.make_seed(error=error)
                generated_text.append(generated_line)
                print(generated_line)
                generated_line = ""
                # return generated_text
            else:
                generated_line += self.decoding[prediction_index] + " "
                X=list(X[0])
                X.append( self.give_vect(
                    self.decoding[prediction_index]))
                X=np.array([X])
                seed_len += 1

        return generated_text

    def make_seed(self, seed_phrase=None, error=0):
        # X = np.zeros((1, self.sentence_length, 300), dtype=np.float32)
        X=[]
        seed_len = 0
        # print(self.decoding[int(rd() * self.num_chars)])
        if not seed_phrase:
            seed_len = randint(1, 5)
            for i in range(seed_len):
                X.append( self.give_vect(
                    self.decoding[int(randint(1, self.num_chars))]))
        else:
            for i, word in enumerate(word_tokenize(seed_pattern)):
                X.append( self.give_vect(word) )
                seed_len += 1
        if error:
            for i in range(seed_len):
                if rd() > 0.5:
                    num_changes = randint(min(1, error - 5), max(1, error))
                    for j in range(num_changes):
                        X[i][j] = X[i][j] + (0.5 - rd()) / 2
        return np.array([X]), seed_len
