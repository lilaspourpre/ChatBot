from keras.layers import Input, Embedding, LSTM, Dense, RepeatVector, Dropout, merge
from keras.optimizers import Adam
from keras.models import Model
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.preprocessing import sequence
from entities.TrainModel import TrainModel
import keras.backend as K
import numpy as np
np.random.seed(1234)  # for reproducibility

class GAN2(TrainModel):
    def __init__(self, vocabulary_size, embedding_size, hidden_size, max_len):
        super(TrainModel, self).__init__()
        ad = Adam(lr=0.00005)

        input_context = Input(shape=(max_len,), dtype='int32', name='the context text')
        input_answer = Input(shape=(max_len,), dtype='int32', name='the answer text up to the current token')
        LSTM_encoder = LSTM(hidden_size, init= 'lecun_uniform', name='Encode context')
        LSTM_decoder = LSTM(hidden_size, init= 'lecun_uniform', name='Encode answer up to the current token')

        Shared_Embedding = Embedding(output_dim=embedding_size, input_dim=vocabulary_size, input_length=max_len, name='Shared')
        word_embedding_context = Shared_Embedding(input_context)
        context_embedding = LSTM_encoder(word_embedding_context)

        word_embedding_answer = Shared_Embedding(input_answer)
        answer_embedding = LSTM_decoder(word_embedding_answer)

        merge_layer = merge([context_embedding, answer_embedding], mode='concat', concat_axis=1, name='concatenate the embeddings of the context and the answer up to current token')
        out = Dense(vocabulary_size/2, activation="relu", name='relu activation')(merge_layer)
        out = Dense(vocabulary_size, activation="softmax", name='likelihood of the current token using softmax activation')(out)

        model = Model(input=[input_context, input_answer], output = [out])

        model.compile(loss='categorical_crossentropy', optimizer=ad)
