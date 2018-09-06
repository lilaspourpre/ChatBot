import argparse
import os
os.environ['KERAS_BACKEND']='tensorflow'
from keras import backend as K
from entities.Model import Model


def __get_external_parameters():
    parser = argparse.ArgumentParser(description='Train sequence model for dialogues')
    parser.add_argument('-c', type=str, dest='config_path', metavar='<config>',
                        required=True, help='config file path')
    parser.add_argument('-m', type=str, dest='model_file', metavar='<model file>',
                        required=True, help='model file path')
    parser.add_argument('-j', type=str, dest='json_config_path', metavar='<json>',
                        required=True, help='json config file path')
    args = parser.parse_args()
    return args.config_path, args.model_file, args.json_config_path


def main():
    config_path, model_path, json_path = __get_external_parameters()
    model = Model(config_path, model_path, json_path)
    # input_phrase = "Какой полк?"
    # print(input_phrase)
    # print(model.answer(input_phrase))
    # input_phrase = "Успеют ли наши?"
    # print(input_phrase)
    # print(model.answer(input_phrase))
    while True:
        question = input("Ваш вопрос: ")
        if question.lower() == "exit" or question.lower() == "выйти":
            print("Пока!")
            break
        try:
            print(model.answer(str(question)))
        except KeyError as e:
            print("Я не знаю слова", e, ",", "простите")
    K.clear_session()


if __name__ == '__main__':
    main()
