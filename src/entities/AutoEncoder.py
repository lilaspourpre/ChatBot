from keras.models import *
from .HelperTransformerClasses import *
from .TrainModel import TrainModel
from keras.preprocessing.sequence import pad_sequences
import numpy as np


class AutoEncoder(TrainModel):
    model_name = "auto"

    def __init__(self, vocabulary_size, embedding_size, hidden_size, max_len, target_directory):
        super(AutoEncoder, self).__init__()
        self.encoder_input = Input(shape=(max_len,))
        self.model = Sequential()
        self.model.add(Embedding(vocabulary_size, embedding_size, input_length=max_len))
        self.model.add(LSTM(hidden_size, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(TimeDistributed(Dense(vocabulary_size, activation='softmax')))

    def train_model(self, dataset, max_len, decode_size, batch_size, epochs):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit_generator(self.__generator(dataset, batch_size, max_len, decode_size), epochs=epochs,
                                 steps_per_epoch=int(len(dataset) / batch_size),
                                 validation_data=self.__generator(dataset, batch_size * 2, max_len, decode_size,
                                                                  save=False),
                                 validation_steps=int(len(dataset) / batch_size * 2))
        return self.model

    def __generator(self, dataset, batch_size, max_len, decode_size, save=True):
        samples_per_epoch = len(dataset)
        number_of_batches = int(samples_per_epoch / batch_size)
        counter = 0
        while 1:
            dataset_batch = np.array(dataset[batch_size * counter:batch_size * (counter + 1)])
            x_batch = pad_sequences([d[0] for d in dataset_batch], maxlen=max_len, padding="post")
            y_batch = pad_sequences([[self.decode(i, decode_size) for i in d[1]] for d in dataset_batch],
                                    maxlen=max_len, padding="post")
            counter += 1
            yield [x_batch], y_batch
            # restart counter to yeild data in the next epoch as well
            if counter == number_of_batches:
                counter = 0


