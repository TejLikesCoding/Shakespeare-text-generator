import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.optimizers import RMSprop

filepath = keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

text = text[300000:800000]

characters = sorted(set(text))

char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))

SEQ_length = 40
STEP_SIZE = 3

#Below is the code used to train my model
'''
sentences = []
next_characters = []


for i in range(0, len(text) - SEQ_length, STEP_SIZE):
    sentences.append(text[i: i + SEQ_length])
    next_characters.append(text[i + SEQ_length])

x = np.zeros((len(sentences), SEQ_length, len(characters)), dtype=np.bool)
y = np.zeros((len(sentences), len(characters)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, character in enumerate(sentence):
        x[i, t, char_to_index[character]] = 1
    y[i, char_to_index[next_characters[i]]] = 1

model = Sequential()
model.add(LSTM(128, input_shape=(SEQ_length, len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))

model.fit(x, y, batch_size=256, epochs=20)

model.save('textgenerator.model1')

'''

model = tf.keras.models.load_model('textgenerator.model1')




def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)/temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

#takes predictions and picks one character depending on temperature, High = risky, Low = Safe


def generate_text(length, temperature):
    start_index = random.randint(0, len(text)-SEQ_length-1)
    generated = ''
    sentence = text[start_index: start_index + SEQ_length]
    generated += sentence 
    for i in range(length):
        x = np.zeros((1, SEQ_length, len(characters)))
        for t, character in enumerate(sentence):
            x[0, t, char_to_index[character]] = 1



        predictions = model.predict(x, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character
    return generated

generated_text = generate_text(length=100, temperature=0.5)
print(generated_text)


