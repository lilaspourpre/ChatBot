from keras.models import *
from ..helpers.HelperTransformerClasses import *
from .TrainModel import TrainModel, WeightsSaver
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os


class Transformer(TrainModel):
    model_name = "transformer"

    def __init__(self, vocabulary_size, embedding_size, hidden_size, max_len, target_directory="", n_head=4, d_k=32,
                 d_v=32, layers=2, dropout=0.1):
        super(Transformer, self).__init__()
        self.decode_model = None
        self.max_len = max_len
        self.embedding_size = embedding_size
        self.encoder_input = Input(shape=(max_len,))
        self.decoder_input = Input(shape=(max_len,))
        self.target_directory = target_directory
        self.epoch = 0
        self.embedded = Embedding(input_dim=vocabulary_size, output_dim=embedding_size, trainable=False)(
            self.encoder_input)

        positional_embeddings = Embedding(vocabulary_size, embedding_size, trainable=False,
                                          weights=[GetPosEncodingMatrix(vocabulary_size, embedding_size)])
        input_word_emb = Embedding(vocabulary_size, embedding_size)
        output_word_emb = input_word_emb

        self.encoder = Encoder(embedding_size, hidden_size, n_head, d_k, d_v, layers, dropout, word_emb=input_word_emb,
                               pos_emb=positional_embeddings)
        self.decoder = Decoder(embedding_size, hidden_size, n_head, d_k, d_v, layers, dropout, word_emb=output_word_emb,
                               pos_emb=positional_embeddings)
        self.target_layer = TimeDistributed(Dense(vocabulary_size, use_bias=False))

        source_position = Lambda(self.__get_pos_seq)(self.encoder_input)
        target_position = Lambda(self.__get_pos_seq)(self.decoder_input)
        enc_output = self.encoder(self.encoder_input, source_position)
        dec_output = self.decoder(self.decoder_input, target_position, self.encoder_input, enc_output)
        final_output = self.target_layer(dec_output)
        self.model = Model([self.encoder_input, self.decoder_input], final_output)

    @staticmethod
    def __get_pos_seq(x):
        mask = K.cast(K.not_equal(x, 0), 'int32')
        pos = K.cumsum(K.ones_like(x, 'int32'), 1)
        return pos * mask

    def train_model(self, dataset, decode_size, batch_size, epochs):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit_generator(self.__generator(dataset, batch_size, self.max_len, decode_size), epochs=epochs,
                                 steps_per_epoch=int(len(dataset) / batch_size),
                                 validation_data=self.__generator(dataset, batch_size * 2, self.max_len, decode_size),
                                 validation_steps=int(len(dataset) / batch_size * 2),
                                 callbacks=[WeightsSaver(self.target_directory, self.model_name)])

    def save_model(self):
        self.model.save_weights(os.path.join(self.target_directory, "{}_final_model.h5".format(self.model_name)))

    def predict(self, x_input):
        x_pad = pad_sequences([x_input], maxlen=self.max_len, padding="post")
        target_seq = np.zeros((1, self.max_len), dtype='int32')
        last_output = None
        for i in range(self.max_len):
            output = self.model.predict_on_batch([x_pad, target_seq])
            sampled_index = np.argmax(output[0, i, :])
            target_seq[0, i] = sampled_index
            last_output = output
        return last_output

    def __generator(self, dataset, batch_size, max_len, decode_size):
        samples_per_epoch = len(dataset)
        number_of_batches = int(samples_per_epoch / batch_size)
        counter = 0
        while 1:
            dataset_batch = np.array(dataset[batch_size * counter:batch_size * (counter + 1)])
            x_batch = pad_sequences([d[0] for d in dataset_batch], maxlen=max_len, padding="post")
            y_batch_input = pad_sequences([d[1] for d in dataset_batch],
                                          maxlen=max_len, padding="post")
            y_batch = pad_sequences([[self.decode(i, decode_size) for i in d[1]] for d in dataset_batch],
                                    maxlen=max_len, padding="post")
            counter += 1
            yield [x_batch, y_batch_input], y_batch
            # restart counter to yeild data in the next epoch as well
            if counter == number_of_batches:
                counter = 0


class Encoder:
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v,
                 layers=6, dropout=0.1, word_emb=None, pos_emb=None):
        self.emb_layer = word_emb
        self.pos_layer = pos_emb
        self.layers = [EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout) for _ in range(layers)]

    def __call__(self, src_seq, src_pos, active_layers=999):
        x = self.emb_layer(src_seq)
        if src_pos is not None:
            pos = self.pos_layer(src_pos)
            x = Add()([x, pos])
        mask = Lambda(lambda x: GetPadMask(x, x))(src_seq)
        for enc_layer in self.layers[:active_layers]:
            x, att = enc_layer(x, mask)
        return x


class Decoder:
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v,
                 layers=6, dropout=0.1, word_emb=None, pos_emb=None):
        self.emb_layer = word_emb
        self.pos_layer = pos_emb
        self.layers = [DecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout) for _ in range(layers)]

    def __call__(self, tgt_seq, tgt_pos, src_seq, enc_output, active_layers=999):
        dec = self.emb_layer(tgt_seq)
        pos = self.pos_layer(tgt_pos)
        x = Add()([dec, pos])
        self_pad_mask = Lambda(lambda x: GetPadMask(x, x))(tgt_seq)
        self_sub_mask = Lambda(GetSubMask)(tgt_seq)
        self_mask = Lambda(lambda x: K.minimum(x[0], x[1]))([self_pad_mask, self_sub_mask])
        enc_mask = Lambda(lambda x: GetPadMask(x[0], x[1]))([tgt_seq, src_seq])
        for dec_layer in self.layers[:active_layers]:
            x, self_att, enc_att = dec_layer(x, enc_output, self_mask, enc_mask)
        return x
