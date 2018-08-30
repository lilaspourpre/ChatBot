import os
import numpy as np
os.environ['KERAS_BACKEND']='tensorflow'
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Input, Embedding, LSTM, Concatenate
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense
from keras.layers.wrappers import Bidirectional
from entities.TrainModel import TrainModel


def __decode(num, decode_size):
    result = [0] * decode_size
    result[num] = 1
    return result


class GAN3(TrainModel):
    def __init__(self, vocabulary_size, embedding_size, hidden_size, max_len, target_directory):
        super(TrainModel, self).__init__()
        self.max_len = max_len
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.OPTIMIZER = Adam(lr=0.0002, decay=8e-9)
        self.noise_gen = np.random.normal(0, 1, (100,))
        self.G = self.generator()
        self.G.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER)
        self.D = self.discriminator()
        self.D.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER, metrics=['accuracy'])
        #self.stacked_G_D = self.stack_G_D()
        #self.stacked_G_D.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER)

    def generator(self):
        input_context = Input(shape=(self.max_len,), dtype='int32', name='thecontexttext')
        lstm_encoder = LSTM(self.hidden_size, init='lecun_uniform', name='Encodecontext')
        shared_embedding = Embedding(output_dim=self.embedding_size, input_dim=self.vocabulary_size,
                                     input_length=self.max_len,
                                     name='Shared')
        word_embedding_context = shared_embedding(input_context)
        context_embedding = lstm_encoder(word_embedding_context)
        out = Dense(int(self.vocabulary_size / 2), activation="relu", name='reluactivation')(context_embedding)
        out = Dense(self.max_len, activation="softmax",
                    name='likelihoodofthecurrenttokenusingsoftmaxactivation')(out)
        model = Model(input=[input_context], output=[out])
        return model

    def discriminator(self):
        input_context = Input(shape=(self.max_len,), dtype='int32', name='input_context')
        lstm_encoder_discriminator = Bidirectional(LSTM(self.hidden_size, init='lecun_uniform'),
                                                   name='encoder_discriminator')
        shared_embedding = Embedding(output_dim=self.embedding_size, input_dim=self.vocabulary_size,
                                     input_length=self.max_len, trainable=False, name='shared')
        word_embedding_context = shared_embedding(input_context)
        context_embedding_discriminator = lstm_encoder_discriminator(word_embedding_context)
        loss = Dense(1, activation="sigmoid", name='discriminator_output')(context_embedding_discriminator)
        model = Model(input=[input_context], output=[loss])
        return model

    def stack_G_D(self):
        pass
        # self.D.trainable = False
        # input_context = Input(shape=(self.max_len,), dtype='int32', name='input_context')
        # input_answer = Input(shape=(self.max_len,), dtype='int32', name='input_answer')
        # dis = self.D([input_context, input_answer])
        # predictions = self.G(dis)
        # model = Model(input=[input_context, input_answer], output=predictions)
        #
        # return model

    def train_model(self, dataset, max_len, decode_size, batch_size=32, epochs=200):
        x_pad = pad_sequences([d[0] for d in dataset], maxlen=max_len, padding="post")
        y_pad = pad_sequences([d[1] for d in dataset], maxlen=max_len,
                              padding="post")

        train_labels = [1 for _ in x_pad]
        for cnt in range(epochs):
            # train discriminator
            d_loss = self.D.train_on_batch([x_pad], train_labels)
            # train generator
            g_loss = self.G.train_on_batch([x_pad], y_pad)
            print('epoch: %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (cnt, d_loss[0], g_loss))
        return self.G

    def __decode(self, num, decode_size):
        result = [0] * decode_size
        result[num] = 1
        return result
