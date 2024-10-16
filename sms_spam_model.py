# -*- coding: utf-8 -*-
"""sms_spam_model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1mnvzFyu_7o9ZmUJbUPB3Mp2NRocTcK9C
"""

# !nvidia-smi

# from google.colab import drive
# drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
import re
import string
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gradio as gr

from tqdm import tqdm
import os
import nltk
import random

import keras
from keras.models import Sequential
from keras.initializers import Constant
from keras.layers import (LSTM,
                          Embedding,
                          BatchNormalization,
                          Dense,
                          TimeDistributed,
                          Dropout,
                          Bidirectional,
                          Flatten,
                          GlobalMaxPool1D,
                          GlobalAveragePooling1D)
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    accuracy_score
)
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

data_path = 'spam.csv'
data = pd.read_csv(data_path, encoding='latin-1')
data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
data.columns = ['label', 'message']
data['message_len'] = data['message'].apply(lambda x:len(x.split(' ')))
balance_count = data.groupby('label')['label'].agg('count').values

def data_cleaning(text):
  text = str(text).lower()
  text = re.sub('\[.*?\]', '', text)
  text = re.sub('https?://\S+|www\.\S+', '', text)
  text = re.sub('<.*?>+', '', text)
  text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
  text = re.sub('\n', '', text)
  text = re.sub('\w*\d\w*', '', text)
  return text

nltk.download('all')
stop_words = set(stopwords.words('english'))
more_stp_words = ['u', 'im', 'c']
stop_words = stop_words.union(more_stp_words)

stemmer = nltk.SnowballStemmer('english')
def stem_words(text):
  return ' '.join([stemmer.stem(word) for word in text.split()])

# from matplotlib import pyplot as plt
# import seaborn as sns
# data.groupby('label').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
# plt.gca().spines[['top', 'right',]].set_visible(False)

def data_preprocessing(text):
  text = data_cleaning(text)
  text = stem_words(text)
  return text

data['message_clean'] = data['message'].apply(lambda x: data_preprocessing(x))
data.head()

le = LabelEncoder()
data['label_encoded'] = le.fit_transform(data['label'])
data.head()

X = data['message_clean']
y = data['label_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

vect = CountVectorizer()
vect.fit(X_train)

X_train_dtm = vect.transform(X_train)
X_test_dtm = vect.transform(X_test)

vect_tunned = CountVectorizer(stop_words='english', ngram_range=(1,2), min_df=0.1, max_df=0.7, max_features=100)

tfidf_transformer = TfidfTransformer()
tfidf_transformer.fit(X_train_dtm)
X_train_tfidf = tfidf_transformer.transform(X_train_dtm)

texts = data['message_clean']
labels = data['label_encoded']

word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(texts)

vocab_length = len(word_tokenizer.word_index) + 1
vocab_length

def embed(corpus):
    return word_tokenizer.texts_to_sequences(corpus)

longest_train = max(texts, key=lambda sentence: len(word_tokenize(sentence)))
length_long_sentence = len(word_tokenize(longest_train))

train_padded_sentences = pad_sequences(
    embed(texts),
    length_long_sentence,
    padding='post'
)

train_padded_sentences

embeddings_dictionary = dict()
embedding_dim = 100
with open('glove.6B.100d.txt') as fp:
    for line in fp.readlines():
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        embeddings_dictionary [word] = vector_dimensions

embedding_matrix = np.zeros((vocab_length, embedding_dim))

for word, index in word_tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

embedding_matrix

new_embedding_dim = 100
svd = TruncatedSVD(n_components=new_embedding_dim, n_iter=7, random_state=42)
reduced_embedding_matrix = svd.fit_transform(embedding_matrix)
reduced_embedding_matrix

cosine_similarity_matrix = cosine_similarity(reduced_embedding_matrix)
cosine_similarity_matrix

X_train, X_test, y_train, y_test = train_test_split(
    train_padded_sentences,
    labels,
    test_size=0.25
)

embedding_matrix.shape

def glove_lstm():
    model = Sequential()

    model.add(Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=embedding_matrix.shape[1],
        weights = [embedding_matrix],
        input_length=length_long_sentence
    ))

    model.add(Bidirectional(LSTM(
        length_long_sentence,
        return_sequences = True,
        recurrent_dropout=0.2
    )))

    model.add(GlobalMaxPool1D())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(length_long_sentence, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(length_long_sentence, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    return model

model = glove_lstm()
model.summary()

model = glove_lstm()

checkpoint = ModelCheckpoint(
    'model.keras',
    monitor = 'val_loss',
    verbose = 1,
    save_best_only = True
)
reduce_lr = ReduceLROnPlateau(
    monitor = 'val_loss',
    factor = 0.2,
    verbose = 1,
    patience = 5,
    min_lr = 0.001
)
history = model.fit(
    X_train,
    y_train,
    epochs = 7,
    batch_size = 32,
    validation_data = (X_test, y_test),
    verbose = 1,
    callbacks = [reduce_lr, checkpoint]
)


# Load your trained model
model = load_model('model.keras')

def preprocess_input(text):
    MAX_LENGTH = 100
    sequence = word_tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LENGTH)
    return padded_sequence

def predict(text):
    processed_input = preprocess_input(text)
    prediction = model.predict(processed_input)
    return True if prediction > 0.5 else False

# Create a Gradio interface
gradio_app = gr.Interface(
    fn=predict,
    inputs="textbox",
    outputs="text",
    title="Spam Detection Model",
    description="Enter a message to check if it's spam or not."
)

# Launch Gradio using CLI
if __name__ == "__main__":
    gradio_app.launch()

