from keras.layers import Input, Embedding, LSTM, Concatenate
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense
from keras.layers.wrappers import Bidirectional
from entities.TrainModel import TrainModel
import numpy as np
np.random.seed(1234)  # for reproducibility


class GAN2(TrainModel):
    def __init__(self, vocabulary_size, embedding_size, hidden_size, max_len):
        super(TrainModel, self).__init__()
        self.ad = Adam(lr=0.00005)
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.max_len = max_len

        input_context = Input(shape=(self.max_len,), dtype='int32', name='the context text')
        input_answer = Input(shape=(self.max_len,), dtype='int32', name='the answer text up to the current token')
        LSTM_encoder = LSTM(self.hidden_size, init='lecun_uniform', name='Encode context')
        LSTM_decoder = LSTM(self.hidden_size, init='lecun_uniform', name='Encode answer up to the current token')

        Shared_Embedding = Embedding(output_dim=self.embedding_size, input_dim=self.vocabulary_size, input_length=self.max_len,
                                     name='Shared')
        word_embedding_context = Shared_Embedding(input_context)
        context_embedding = LSTM_encoder(word_embedding_context)

        word_embedding_answer = Shared_Embedding(input_answer)
        answer_embedding = LSTM_decoder(word_embedding_answer)

        merge_layer = Concatenate([context_embedding, answer_embedding], axis=1,
                                  name='concatenate the embeddings of the context and the answer up to current token')
        out = Dense(self.vocabulary_size / 2, activation="relu", name='relu activation')(merge_layer)
        out = Dense(self.vocabulary_size, activation="softmax",
                    name='likelihood of the current token using softmax activation')(out)

        model = Model(input=[input_context, input_answer], output=[out])
        model.compile(loss='categorical_crossentropy', optimizer=self.ad)

    def discriminator_model(self, learning_rate=0.005, train_mode=True):
        ad = Adam(lr=learning_rate)

        input_context = Input(shape=(self.max_len,), dtype='int32', name='input context')
        input_answer = Input(shape=(self.max_len,), dtype='int32', name='input answer')
        input_current_token = Input(shape=(self.vocabulary_size,), name='input_current_token')

        LSTM_encoder_discriminator = Bidirectional(LSTM(self.hidden_size, init='lecun_uniform'),
                                                       name='encoder discriminator')
        LSTM_decoder_discriminator = LSTM(self.hidden_size, init='lecun_uniform', name='decoder discriminator')

        if os.path.isfile(weights_file):
            Shared_Embedding = Embedding(output_dim=self.embedding_size, input_dim=self.vocabulary_size,
                                         input_length=self.max_len, trainable=False, name='shared')
        else:
            Shared_Embedding = Embedding(output_dim=self.embedding_size, input_dim=self.vocabulary_size,
                                         weights=[embedding_matrix], input_length=self.max_len, trainable=False,
                                         name='shared')

        word_embedding_context = Shared_Embedding(input_context)
        word_embedding_answer = Shared_Embedding(input_answer)
        context_embedding_discriminator = LSTM_encoder_discriminator(word_embedding_context)
        answer_embedding_discriminator = LSTM_decoder_discriminator(word_embedding_answer)
        loss = Concatenate([context_embedding_discriminator, answer_embedding_discriminator, input_current_token],
                     axis=1, name='concatenation discriminator')
        loss = Dense(1, activation="sigmoid", name='discriminator output')(loss)

        model = Model(input=[input_context, input_answer, input_current_token], output=[loss])

        if train_mode:
            model.compile(loss='binary_crossentropy', optimizer=ad)
        else:
            model.compile(loss='negative_crossentropy', optimizer=ad)

        return model

    def concat_generator_and_discriminator(self,  learning_rate=0.005):
        ad = Adam(lr=learning_rate)

        input_context = Input(shape=(self.max_len,), dtype='int32', name='input context')
        input_answer = Input(shape=(self.max_len,), dtype='int32', name='input answer')
        input_current_token = Input(shape=(self.vocabulary_size,), name='input_current_token')

        LSTM_encoder_discriminator = Bidirectional(LSTM(self.hidden_size, init='lecun_uniform'),
                                                       name='encoder discriminator')

        LSTM_decoder_discriminator = LSTM(self.hidden_size, init='lecun_uniform', name='decoder discriminator')
        Shared_Embedding = Embedding(output_dim=self.embedding_size, input_dim=self.vocabulary_size,
                                     input_length=self.max_len,
                                     trainable=False, name='shared')
        word_embedding_context = Shared_Embedding(input_context)
        word_embedding_answer = Shared_Embedding(input_answer)
        context_embedding_discriminator = LSTM_encoder_discriminator(word_embedding_context)
        answer_embedding_discriminator = LSTM_decoder_discriminator(word_embedding_answer)
        loss = Concatenate([context_embedding_discriminator, answer_embedding_discriminator, input_current_token],
                            axis=1, name='concatenation discriminator')
        loss = Dense(1, activation="sigmoid", name='discriminator output')(loss)
        model = Model(input=[input_context, input_answer, input_current_token], output=[loss])
        model.compile(loss='binary_crossentropy', optimizer=ad)

        return model