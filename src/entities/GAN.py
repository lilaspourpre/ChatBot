import os
import numpy as np
os.environ['KERAS_BACKEND']='tensorflow'
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam
from entities.TrainModel import TrainModel



def __decode(num, decode_size):
    result = [0] * decode_size
    result[num] = 1
    return result


class GAN(TrainModel):
    def __init__(self, vocabulary_size, embedding_size, hidden_size, max_len, channels=1):
        super(TrainModel, self).__init__()
        self.WIDTH = max_len
        self.HEIGHT = vocabulary_size
        self.CHANNELS = channels
        self.SHAPE = (self.WIDTH, self.HEIGHT, self.CHANNELS)
        self.OPTIMIZER = Adam(lr=0.0002, decay=8e-9)
        self.noise_gen = np.random.normal(0, 1, (100,))
        self.G = self.generator()
        self.G.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER)
        self.D = self.discriminator()
        self.D.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER, metrics=['accuracy'])
        self.stacked_G_D = self.stack_G_D()
        self.stacked_G_D.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER)

    def generator(self):
        model = Sequential()
        model.add(Dense(256, input_shape=(100,)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.WIDTH * self.HEIGHT * self.CHANNELS, activation='tanh'))
        model.add(Reshape((self.WIDTH, self.HEIGHT, self.CHANNELS)))

        return model

    def discriminator(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.SHAPE))
        model.add(Dense((self.WIDTH * self.HEIGHT * self.CHANNELS), input_shape=self.SHAPE))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(int((self.WIDTH * self.HEIGHT * self.CHANNELS) / 2)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        return model

    def stack_G_D(self):
        self.D.trainable = False
        model = Sequential()
        model.add(self.G)
        model.add(self.D)
        return model

    def train_model(self, dataset, max_len, decode_size, batch_size=32, epochs=200):
        for cnt in range(epochs):
            # train discriminator
            random_index = np.random.randint(0, len(X_train) - int(batch_size / 2))
            legit_images = X_train[random_index: random_index + int(batch_size / 2)].reshape(int(batch_size / 2),
                                                                                             self.WIDTH, self.HEIGHT,
                                                                                             self.CHANNELS)
            gen_noise = np.random.normal(0, 1, (int(batch_size / 2), 100))
            syntetic_images = self.G.predict(gen_noise)
            x_combined_batch = np.concatenate((legit_images, syntetic_images))
            y_combined_batch = np.concatenate((np.ones((int(batch_size / 2), 1)), np.zeros((int(batch_size / 2), 1))))
            d_loss = self.D.train_on_batch(x_combined_batch, y_combined_batch)
            # train generator
            noise = np.random.normal(0, 1, (batch_size, 100))
            y_mislabled = np.ones((batch_size, 1))
            g_loss = self.stacked_G_D.train_on_batch(noise, y_mislabled)
            print('epoch: %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (cnt, d_loss[0], g_loss))

#np.expand_dims(X_train, axis=3)
