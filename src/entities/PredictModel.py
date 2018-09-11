import numpy as np
from nltk.tokenize import word_tokenize
import re


class PredictModel:
    def __init__(self, model, parameters):
        self.__model = model
        self.__id2word, self.__word2id, self.__max_len = parameters

    def answer(self, raw_text_question):
        encoded_question = self.__encode(raw_text_question)
        predicted = self.__model.predict(encoded_question)
        answer = [self.__id2word[str(np.argmax(predicted[i][j]))] for i in range(len(predicted))
                  for j in range(len(predicted[i]))]
        result_text = " ".join(answer).capitalize()
        return re.sub("( ,)+", ",", re.sub("( !)+", "!", re.sub("( \?)+", "?", re.sub("( \.)+", ".", result_text))))

    def __encode(self, input_phrase):
        tokenized = word_tokenize(input_phrase, "russian")
        sentence_int = [self.__word2id[w.lower()] for w in tokenized]
        return sentence_int
