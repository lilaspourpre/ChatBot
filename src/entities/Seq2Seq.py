from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.layers import LSTM, Bidirectional, Input, TimeDistributed, Dense, RepeatVector, Flatten, Activation, Permute


class CustomSeq2Seq:
    def __init__(self, vocabulary_size, embedding_size, hidden_size, max_len):
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
        # decoder step
        # TODO: add decoder input
        self.decoder = Bidirectional(LSTM(units=hidden_size, input_shape=(max_len, embedding_size),
                                          return_sequences=True, ))(self.attention)
        self.output = TimeDistributed(Dense(vocabulary_size, activation="softmax"))(self.decoder)
        self.model = Model(inputs=self.encoder_input, outputs=self.output)



































        # def __init__(self, encoder_input, encoder_input_length,
        #              decoder_input, decoder_input_length,
        #              expected_decoder_output, expected_decoder_output_length,
        #              embedding_size,
        #              vocabulary_size,
        #              hidden_size,
        #              learning_rate=0.0005,
        #              max_gradient_norm=1,
        #              offset=1):
        #     # params
        #     self.max_gradient_norm = max_gradient_norm
        #     self.learning_rate = tf.placeholder_with_default(learning_rate, [], 'learning_rate')
        #     self.keep_prob = tf.placeholder_with_default(1.0, [], "keep_prob")
        #     self.batch_size = tf.shape(self.encoder_input)[0]
        #     self.hidden_size = hidden_size
        #
        #     # inputs
        #     self.encoder_input = tf.identity(encoder_input, 'encoder_input')
        #     self.encoder_input_length = tf.identity(encoder_input_length, 'encoder_input_length')
        #
        #     self.decoder_input = tf.identity(decoder_input, 'decoder_input')
        #     self.decoder_input_length = tf.identity(decoder_input_length, 'decoder_input_length')
        #
        #     self.expected_decoder_output = tf.identity(expected_decoder_output, 'expected_decoder_output')
        #     self.expected_decoder_output_length = tf.identity(expected_decoder_output_length,
        #                                                       'expected_decoder_output_length')
        #
        #     # embeddings
        #     self.input_embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -0.03, 0.03))
        #     self.encoder_embeddings = tf.nn.embedding_lookup(self.input_embeddings, self.encoder_input, max_norm=1.0)
        #
        #     # encoder
        #     encoder_outputs, encoder_state = self.__add_encoder(self.encoder_embeddings, self.encoder_input_length,
        #                                                         self.keep_prob)
        #     encoder_outputs = tf.concat([encoder_outputs, self.encoder_input], -1)
        #
        #     # decoder
        #     self.outputs, self.decoder_cell, self.decoder_state, self.lstm_to_output_layer = self._add_train_decoder(
        #         encoder_outputs, encoder_state)
        #
        #     # update
        #     self.loss = self.__add_loss(self.outputs)
        #     self.update_step = self.__add_update_step(self.loss)
        #     self.translations, self.attention, self.translations_lengths = self.__add_inference(self.decoder_cell,
        #                                                                                         self.decoder_state,
        #                                                                                         self.lstm_to_output_layer)
        #
        # def __add_encoder(self, encoder_input, encoder_input_length, keep_prob):
        #     forward_encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
        #     backward_encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
        #
        #     encoder_bi_outputs, (forward_encoder_state, backward_encoder_state) = tf.nn.bidirectional_dynamic_rnn(
        #         forward_encoder_cell, backward_encoder_cell, encoder_input, sequence_length=encoder_input_length,
        #         dtype=tf.float32)
        #
        #     encoder_outputs = tf.concat(encoder_bi_outputs, -1)
        #     encoder_outputs = tf.nn.dropout(encoder_outputs, keep_prob)
        #     encoder_state = tf.contrib.rnn.LSTMStateTuple(
        #         tf.nn.dropout(tf.concat([forward_encoder_state.c, backward_encoder_state.c], -1), keep_prob),
        #         tf.nn.dropout(tf.concat([forward_encoder_state.h, backward_encoder_state.h], -1), keep_prob))
        #
        #     return encoder_outputs, encoder_state
        #
        # def _add_train_decoder(self, a, d):
        #     return a, a, a, a
        #
        # def __add_loss(self, outputs):
        #     mask = tf.sequence_mask(
        #         self.expected_decoder_output_length,
        #         dtype=tf.float32)
        #
        #     losses = tf.squeeze(
        #         tf.losses.cosine_distance(
        #             tf.nn.l2_normalize(self.expected_decoder_output, -1),
        #             outputs,
        #             axis=-1,
        #             weights=1.0,
        #             reduction=tf.losses.Reduction.NONE),
        #         [-1])
        #     loss = tf.reduce_sum(losses * mask) / tf.cast(tf.reduce_sum(self.expected_decoder_output_length), tf.float32)
        #     return loss
        #
        # def __add_update_step(self, loss):
        #     params = tf.trainable_variables()
        #     gradients = tf.gradients(loss, params)
        #     clipped_gradients, self.gradients_norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        #     optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        #     update_step = optimizer.apply_gradients(zip(clipped_gradients, params))
        #     return update_step
