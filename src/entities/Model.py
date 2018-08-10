import numpy as np
from nltk.tokenize import word_tokenize
from keras import models
from keras.preprocessing.sequence import pad_sequences

from main import load_config


class Model:
    def __init__(self, config_path, model_path):
        self.__model = models.load_model(model_path)
        self.__id2word, self.__word2id, self.__max_len = load_config(config_path)

    def answer(self, raw_text_question):
        encoded_question = self.__encode(raw_text_question)
        answer = self.__predict(encoded_question)
        return " ".join(answer).capitalize()

    def __encode(self, input_phrase):
        tokenized = word_tokenize(input_phrase, "english")
        sentence_int = [self.__word2id[w.lower()] for w in tokenized]
        return sentence_int

    def __predict(self, x_input):
        ### TODO: no need to pad
        x_pad = pad_sequences([x_input], maxlen=self.__max_len, padding="post")
        pred = self.__model.predict(x_pad)
        return [self.__id2word[str(np.argmax(pred[i][j]))] for i in range(len(pred)) for j in range(len(pred[i]))]

