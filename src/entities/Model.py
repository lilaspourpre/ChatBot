import numpy as np
from nltk.tokenize import word_tokenize
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
import re
from main import load_config


class Model:
    def __init__(self, config_path, model_path, json_path):
        self.__model = self.__load_model(model_path, json_path)
        self.__id2word, self.__word2id, self.__max_len = load_config(config_path)

    def __load_model(self, weights_path, json_path):
        with open(json_path, "r") as j:
            json_model = j.read()
            model = model_from_json(json_model)
        model.load_weights(weights_path)
        return model

    def answer(self, raw_text_question):
        encoded_question = self.__encode(raw_text_question)
        answer = self.__predict(encoded_question)
        result_text = " ".join(answer).capitalize()
        return re.sub("( ,)+", ",", re.sub("( !)+", "!", re.sub("( \?)+","?", re.sub("( \.)+",".", result_text))))

    def __encode(self, input_phrase):
        tokenized = word_tokenize(input_phrase, "russian")
        sentence_int = [self.__word2id[w.lower()] for w in tokenized]
        return sentence_int

    def __predict(self, x_input):
        x_pad = pad_sequences([x_input], maxlen=self.__max_len, padding="post")
        predicted = self.__model.predict(x_pad)
        return [self.__id2word[str(np.argmax(predicted[i][j]))] for i in range(len(predicted))
                for j in range(len(predicted[i]))]
