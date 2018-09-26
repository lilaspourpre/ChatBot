import os
import numpy as np
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras import backend as K
from entities.train_models.Seq2Seq import CustomSeq2Seq
from entities.train_models.Transformer import Transformer
from entities.train_models.AutoEncoder import AutoEncoder
from entities.train_models.Seq2SeqWithInputs import Seq2Seq
from dataset_creator import *
from entities.PredictModel import PredictModel
from keras.preprocessing.sequence import pad_sequences


def __get_external_parameters():
    parser = argparse.ArgumentParser(description='Train sequence model for dialogues')
    parser.add_argument('-config', type=str, dest='config_file', metavar='<config file>',
                        required=False, help='configuration file with dicts and max_len', default=None)
    parser.add_argument('-model-type', type=choose_model, dest='model', metavar='<model>',
                        required=True, choices=(CustomSeq2Seq, Transformer, Seq2Seq),
                        help='model to use: seq2seq or seq2seqwithinputs or transformer', default="seq2seq")
    parser.add_argument('-embedding', type=int, dest='embedding_size', metavar='<embedding size>',
                        required=False, help='embedding size', default=128)
    parser.add_argument('-hidden', type=int, dest='hidden_size', metavar='<hidden size>',
                        required=False, help='hidden size', default=128)

    subparsers = parser.add_subparsers(help='sub-command help')
    parser_train = subparsers.add_parser('train', help="train mode")
    parser_train.add_argument('-dir', type=str, dest='output_directory', metavar='<directory>', required=True,
                              help='directory for results')
    parser_train.add_argument('-input', type=str, dest='input_file', metavar='<input file>', required=True,
                              help='input file (raw or prepared json)')
    parser_train.add_argument('-epochs', type=int, dest='epochs', metavar='<epochs>', required=False,
                              help='number of epochs', default=500)
    parser_train.add_argument('-batch', type=int, dest='batch_size', metavar='<batch size>', required=False,
                              help='batch size', default=16)
    parser_train.set_defaults(command=_train_mode)

    parser_test = subparsers.add_parser('test', help='test mode')
    parser_test.add_argument('-model-path', type=str, dest='model_path', metavar='<model path>', required=True,
                             help='model path')
    parser_test.set_defaults(command=_test_mode)

    args = parser.parse_args()
    if not hasattr(args, 'command'):
        parser.error('run mode (train or test) not specified')
    return args


def choose_model(encoder_type):
    if encoder_type.lower() == "seq2seq":
        return CustomSeq2Seq
    elif encoder_type.lower() == "transformer":
        return Transformer
    elif encoder_type.lower() == "seq2seqwithinputs":
        return Seq2Seq
    else:
        raise Exception("Unknown encoder name {}".format(encoder_type))


def decode(num, decode_size):
    result = [0] * decode_size
    result[num] = 1
    return result


def read_json_file(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        sentences_list = f.readlines()
    return [json.loads(sentence) for sentence in sentences_list[:-1]]


def __generator(dataset, batch_size, max_len, decode_size):
    def generator():
        samples_per_epoch = len(dataset)
        number_of_batches = int(samples_per_epoch / batch_size)
        counter = 0
        while 1:
            dataset_batch = np.array(dataset[batch_size * counter:batch_size * (counter + 1)])
            x_batch = pad_sequences([d[0] for d in dataset_batch], maxlen=max_len, padding="post")
            y_batch_input = pad_sequences([d[1] for d in dataset_batch],
                                          maxlen=max_len, padding="post")
            y_batch = pad_sequences([[decode(i, decode_size) for i in d[1]] for d in dataset_batch],
                                    maxlen=max_len, padding="post")
            counter += 1
            yield [x_batch, y_batch_input], y_batch
            # restart counter to yeild data in the next epoch as well
            if counter == number_of_batches:
                counter = 0
    return generator


def __generate_no_batches(input_file, max_len, decode_size):
    def generator():
        while 1:
            with open(input_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = json.loads(line)
                    x = pad_sequences([line[0]], maxlen=max_len, padding="post")
                    y = pad_sequences([[decode(i, decode_size) for i in line[1]]], maxlen=max_len, padding="post")
                    yield [x], y
    return generator


def _make_gen(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024*1024)


def rawgencount(filename):
    f = open(filename, 'rb')
    f_gen = _make_gen(f.raw.read)
    return sum(buf.count(b'\n') for buf in f_gen )


def generate_input(args, create_function):
    if args.config_file:
        id2word, word2id, max_len = load_config(args.config_file)
        max_features = len(id2word)
        generator = __generate_no_batches(args.input_file, max_len, max_features)
        return generator, rawgencount(args.input_file), max_len, max_features, 1
    else:
        sentences = read_txt_file(args.input_file)
        dataset, id2word, word2id, max_len = prepare_for_dataset(sentences, create_function)
        write_to_json_config(os.path.join(args.output_directory, "config.json"), id2word, word2id, max_len)
        max_features = len(id2word)
        generator = __generator(dataset, args.batch_size, max_len, max_features)
        return generator, len(dataset), max_len, max_features, args.batch_size


def load_config(config_file):
    config = json.load(open(config_file, 'r'))
    return config["id2word"], config["word2id"], config["max_len"]


def _train_mode():
    def _run(args):
        generator, dataset_len, max_len, max_features, batch_size = generate_input(args, create_dataset)
        nn = args.model(max_features, args.embedding_size, args.hidden_size, max_len, args.output_directory)
        nn.train_model(generator, batch_size, args.epochs, dataset_len)
        nn.save_model()

    return _run


def _test_mode():
    def _run(args):
        if args.config_file:
            id2word, word2id, max_len = load_config(args.config_file)
        else:
            raise FileNotFoundError("Config path is none")
        nn = args.model(len(id2word), args.embedding_size, args.hidden_size, max_len)
        nn.model.load_weights(args.model_path)
        _predict_loop(PredictModel(nn, (id2word, word2id, max_len)))

    return _run


def _predict_loop(model):
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


def main():
    args = __get_external_parameters()
    run_mode = args.command()
    run_mode(args)


if __name__ == '__main__':
    main()
