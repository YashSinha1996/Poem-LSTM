# Prose-LSTM

![Robot Shakespeare](https://vivshaw.github.io/images/robot-shakespeare-teaser.png)

a Keras neural network trained to write ~Shakespearean sonnets~ SO's poems. Letting shakespear name be, because I had got the original idea from it.

## Diffrences from original
This version by me implements a word-level network instead of charecter level network. I did this because I noticed that the orignal word-level generator was generating senseless words when given some proper poems.

To tackle this, the current model takes word2vec vector sequence as input and tries to generate the next words accrding to the current sequence, instead of doing the same at a charecter level.

## Requirements

```
pip install tensorflow
pip install keras
pip install h5py
pip install Flask
pip install Flask-wtf
pip install gunicorn
pip install gensim
```
Download pre-trained word2vec from google
## Training the network

```
python network/train_words.py
```

The weights will be checkpointed as hdf5 files with the format `weights-{epoch:02d}-{loss:.3f}.hdf5` and the model will be dumped as `model.yaml`. If you wish to use a different corpus, just drop it in & edit `network/train.py`.

## Generating text
Edit `network/generate_words.py` to use your new weights and model if desired, then:

```
python network/generate.py
```

If you wish to use different weights and model than I did, put them in `app/static/model.yaml` and `app/static/weights.hdf5`

## Heroku deployment

Should be as easy as:

```
heroku create
git push heroku master
```

You may need to `heroku ps:scale web=1` if it doesn't do so automatically.
