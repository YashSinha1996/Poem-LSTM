import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Dropout, TimeDistributed
from keras.callbacks import ModelCheckpoint
from nltk.tokenize import word_tokenize
import gensim
import progressbar
import pickle

print(Dropout)



def to_base(n,b=5):
    s = ""
    while n>0:
        s = str(n % b) + s
        n /= b
    return s



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
sent_lens={}
X_data, encoding, decoding, max_sent_len = read_file()
try:
    sent_lens = pickle.load(open("sent_lens.pck","rb"))
except Exception as e:
    print(len(encoding))
    print('loading word2vec')
    model = gensim.models.KeyedVectors.load_word2vec_format(
            "/home/yash/Downloads/Work/GoogleNews-vectors-negative300.bin", binary=True)
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
    base=5
    max_len=len(to_base(1300,base))
    print("Vectoring X and y...")
    for i, sentence in enumerate(X_data):
        if not sentence:
            num_sentences -= 1
            continue
        for t, word in enumerate(sentence):
            # X_t = np.zeros((sentence_length, 300), dtype=np.float32)
            # y_t = np.zeros((len(encoding)), dtype=np.bool)
            y_t=0
            X_t = np.array([give_vect(word) for word in sentence[:t + 1]])
            if t + 1 < len(sentence):
                y_t = to_base(encoding[word]).zfill(max_len)
            else:
                y_t = to_base(len(encoding)).zfill(max_len)
            X.append(X_t)
            # print(X_t.shape)
            # exit()
            y.append(np.array([int(a) for a in y_t]))
            num_sentences += 1

    # X = np.array(X)
    # y = np.array(y)
    # print(X.shape,y.shape)

    for i,data in enumerate(X):
        if data.shape[0] in sent_lens:
            sent_lens[data.shape[0]][0].append(data)
            sent_lens[data.shape[0]][1].append(y[i])
        else:
            sent_lens[data.shape[0]]=[[data],[y[i]]]

    with open("sent_lens.pck","wb") as sent_lens_file:
        pickle.dump(sent_lens,sent_lens_file)

# Define our model
print("Let's build a brain!")
model = Sequential()
# model.add(TimeDistributed(Dense(50,activation='relu'), input_shape=(sentence_length, 300)))
model.add(LSTM(128,return_sequences=True, input_shape=(None, 300)))
model.add(Dropout(0.1))
model.add(LSTM(64, return_sequences=True))
# model.add(Dropout(0.1))
model.add(Dropout(0.1))
# model.add(TimeDistributed(Dense(64)))
model.add(LSTM(32))
model.add(Dropout(0.1))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(4))
model.add(Activation('relu'))
model.compile(loss='mean_squared_error', optimizer='adam')

# Dump our model architecture to a file so we can load it elsewhere
architecture = model.to_yaml()
with open('model_words_2.yaml', 'w') as model_file:
    model_file.write(architecture)

# Set up checkpoints
file_path = "weights_words_check_best.hdf5"
checkpoint = ModelCheckpoint(
    file_path, monitor="loss", verbose=0, save_best_only=True, mode="min")
callbacks = [checkpoint]

# Action time! [Insert guitar solo here]
n_epochs=5
epoch_loss=None
new_epoch_loss=None

def print_loss(history):
    print(new_epoch_loss,end="-")

for i in range(n_epochs):
    print("Epoch {}".format(i))
    print("Training =>")
    bar = progressbar.ProgressBar(max_value=len(sent_lens))
    historys=[]
    for n_batch, bath_val in bar(enumerate(sent_lens.values())):
        x = np.array(bath_val[0])
        label = np.array(bath_val[1])
        if x.shape[1] > 1:
            history = model.fit(x,label,verbose=0, callbacks=callbacks.append(print_loss))
            historys.append(min(history.history['loss']))
            if historys:
                new_epoch_loss=sum(historys)/len(historys)
                print(new_epoch_loss)
    print("Loss this epoch: ",new_epoch_loss)
    if not epoch_loss or new_epoch_loss < epoch_loss:
        print("Improved from",epoch_loss)
        epoch_loss = new_epoch_loss
        model.save("weights_words_check_2.hdf5")
