from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.layers import LSTM, Bidirectional, Input, TimeDistributed, Dense, RepeatVector, Flatten, Activation, Permute
from keras.preprocessing.sequence import pad_sequences
from entities.TrainModel import TrainModel


class CustomSeq2Seq(TrainModel):
    def __init__(self, vocabulary_size, embedding_size, hidden_size, max_len):
        super(CustomSeq2Seq, self).__init__()
        self.encoder_input = Input(shape=(max_len,))
        # encoder step
        self.embedded = Embedding(input_dim=vocabulary_size, output_dim=embedding_size, trainable=False)(
            self.encoder_input)
        self.encoder = Bidirectional(
            LSTM(units=hidden_size, input_shape=(max_len, embedding_size), return_sequences=True,
                 dropout=0.25, recurrent_dropout=0.25))(self.embedded)
        # compute importance for each step
        self.attention = Dense(1, activation='tanh')(self.encoder)
        self.attention = Flatten()(self.attention)
        self.attention = Activation('softmax')(self.attention)
        self.attention = RepeatVector(max_len)(self.attention)
        self.attention = Permute([2, 1])(self.attention)
        self.decoder = Bidirectional(LSTM(units=hidden_size, input_shape=(max_len, embedding_size),
                                          return_sequences=True, ))(self.attention)
        self.output = TimeDistributed(Dense(vocabulary_size, activation="softmax"))(self.decoder)
        self.model = Model(inputs=self.encoder_input, outputs=self.output)

    def train_model(self, dataset, max_len, decode_size, batch_size, epochs):
        x_pad = pad_sequences([d[0] for d in dataset], maxlen=max_len, padding="post")
        y_pad = pad_sequences([[self.__decode(i, decode_size) for i in d[1]] for d in dataset], maxlen=max_len,
                              padding="post")
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(x_pad, y_pad, batch_size=batch_size, epochs=epochs, validation_split=0.1)
        return self.model

    def __decode(self, num, decode_size):
        result = [0] * decode_size
        result[num] = 1
        return result