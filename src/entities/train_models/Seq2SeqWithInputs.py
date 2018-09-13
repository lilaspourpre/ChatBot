from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.layers import LSTM, Bidirectional, Input, TimeDistributed, Dense, RepeatVector, Flatten, Activation, \
    Permute, Concatenate
from .TrainModel import TrainModel, WeightsSaver
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os


class Seq2Seq(TrainModel):
    model_name = "seq2seqwithinputs"

    def __init__(self, vocabulary_size, embedding_size, hidden_size, max_len, target_directory=""):
        super(Seq2Seq, self).__init__()
        self.max_len = max_len
        self.encoder_input = Input(shape=(max_len,))
        self.decoder_input = Input(shape=(max_len,))
        self.target_directory = target_directory
        self.epoch = 0
        self.embedded = Embedding(input_dim=vocabulary_size, output_dim=embedding_size, trainable=False)(
            self.encoder_input)
        self.embedded_decoder = Embedding(input_dim=vocabulary_size, output_dim=embedding_size, trainable=False)(
            self.decoder_input)
        self.encoder = Bidirectional(
            LSTM(units=hidden_size, input_shape=(max_len, embedding_size), return_sequences=True,
                 dropout=0.25, recurrent_dropout=0.25))(self.embedded)
        self.attention = Dense(1, activation='tanh')(self.encoder)
        self.attention = Flatten()(self.attention)
        self.attention = Activation('softmax')(self.attention)
        self.attention = RepeatVector(max_len)(self.attention)
        self.attention = Permute([2, 1])(self.attention)
        self.merge = Concatenate(axis=-1)([self.attention, self.embedded_decoder])
        self.decoder = Bidirectional(LSTM(units=hidden_size, input_shape=(max_len, embedding_size),
                                          return_sequences=True))(self.merge)
        self.output = TimeDistributed(Dense(vocabulary_size, activation="softmax"))(self.decoder)
        self.model = Model(inputs=[self.encoder_input, self.decoder_input], outputs=self.output)

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
        target_seq = np.zeros((1, self.max_len), dtype='int32')
        return self.model.predict([x_pad, target_seq])
