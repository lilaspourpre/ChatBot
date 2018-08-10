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
    args = parser.parse_args()
    return args.config_path, args.model_file


def main():
    config_path, model_path = __get_external_parameters()
    model = Model(config_path, model_path)
    input_phrase = "Why?"
    print(input_phrase)
    print(model.answer(input_phrase))
    # while True:
    #     question = input("Your turn: ")
    #     if question.lower() == "exit":
    #         print("Bye!")
    #         break
    #     print("Bot answers:" + model.answer(str(question)))
    K.clear_session()


if __name__ == '__main__':
    main()
