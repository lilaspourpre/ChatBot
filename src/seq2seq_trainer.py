from keras.preprocessing.sequence import pad_sequences
import numpy as np
from entities.dataset import Dataset


def train_model(nn, dataset, max_len, decode_size, epochs):
    x_pad = pad_sequences([d[0] for d in dataset], maxlen=max_len, padding="post")
    y_pad = pad_sequences([[_decode(i, decode_size) for i in d[1]] for d in dataset], maxlen=max_len, padding="post")
    nn.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    generator = Dataset(x_pad, y_pad)
    #TODO: pad problem
    nn.model.fit_generator(generator.get_generator(), steps_per_epoch=1, epochs=epochs)
    return nn.model


def _decode(num, decode_size):
    result = [0]*decode_size
    result[num] = 1
    return result

