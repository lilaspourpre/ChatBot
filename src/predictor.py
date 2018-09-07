from keras import backend as K


def predict_loop(model):
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
