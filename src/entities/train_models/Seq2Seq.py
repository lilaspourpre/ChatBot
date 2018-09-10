from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.layers import LSTM, Bidirectional, Input, TimeDistributed, Dense, RepeatVector, Flatten, Activation, Permute
from .TrainModel import TrainModel, WeightsSaver
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os


class CustomSeq2Seq(TrainModel):
    model_name = "seq2seq"

    def __init__(self, vocabulary_size, embedding_size, hidden_size, max_len, target_directory=""):
        super(CustomSeq2Seq, self).__init__()
        self.max_len = max_len
        self.encoder_input = Input(shape=(max_len,))
        self.target_directory = target_directory
        self.epoch = 0
        self.embedded = Embedding(input_dim=vocabulary_size, output_dim=embedding_size, trainable=False)(
            self.encoder_input)
        self.encoder = Bidirectional(
            LSTM(units=hidden_size, input_shape=(max_len, embedding_size), return_sequences=True,
                 dropout=0.25, recurrent_dropout=0.25))(self.embedded)
        self.attention = Dense(1, activation='tanh')(self.encoder)
        self.attention = Flatten()(self.attention)
        self.attention = Activation('softmax')(self.attention)
        self.attention = RepeatVector(max_len)(self.attention)
        self.attention = Permute([2, 1])(self.attention)
        self.decoder = Bidirectional(LSTM(units=hidden_size, input_shape=(max_len, embedding_size),
                                          return_sequences=True, ))(self.attention)
        self.output = TimeDistributed(Dense(vocabulary_size, activation="softmax"))(self.decoder)
        self.model = Model(inputs=self.encoder_input, outputs=self.output)

    def train_model(self, dataset, decode_size, batch_size, epochs):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit_generator(self.__generator(dataset, batch_size, self.max_len, decode_size), epochs=epochs,
                                 steps_per_epoch=int(len(dataset)/batch_size),
                                 validation_data=self.__generator(dataset, batch_size*2, self.max_len, decode_size),
                                 validation_steps=int(len(dataset)/batch_size*2),
                                 callbacks=[WeightsSaver(self.target_directory, self.model_name)])

    def save_model(self):
        self.model.save_weights(os.path.join(self.target_directory, "{}_final_model.h5".format(self.model_name)))

    def predict(self, x_input):
        x_pad = pad_sequences([x_input], maxlen=self.max_len, padding="post")
        return self.model.predict(x_pad)

    def __generator(self, dataset, batch_size, max_len, decode_size):
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
