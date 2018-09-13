from keras.models import *
from .TrainModel import TrainModel, WeightsSaver
from keras.layers import LSTM, Input, TimeDistributed, Dense, Embedding, Dropout
from keras.preprocessing.sequence import pad_sequences
import os


class AutoEncoder(TrainModel):
    model_name = "autoencoder"

    def __init__(self, vocabulary_size, embedding_size, hidden_size, max_len, target_directory=""):
        super(AutoEncoder, self).__init__()
        self.max_len = max_len
        self.target_directory = target_directory
        self.encoder_input = Input(shape=(max_len,))
        self.model = Sequential()
        self.model.add(Embedding(vocabulary_size, embedding_size, input_length=max_len))
        self.model.add(LSTM(hidden_size, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(TimeDistributed(Dense(vocabulary_size, activation='softmax')))

    def train_model(self, generator, batch_size, epochs, dataset_len):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit_generator(generator(), epochs=epochs,
                                 steps_per_epoch=dataset_len / batch_size,
                                 validation_data=generator(),
                                 validation_steps=dataset_len / batch_size * 2,
                                 callbacks=[WeightsSaver(self.target_directory, self.model_name)])

    def save_model(self):
        self.model.save_weights(os.path.join(self.target_directory, "{}_final_model.h5".format(self.model_name)))

    def predict(self, x_input):
        x_pad = pad_sequences([x_input], maxlen=self.max_len, padding="post")
        return self.model.predict(x_pad)
