import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import argparse
from entities.train_models.AutoEncoder import AutoEncoder
from entities.PredictModel import PredictModel
from main import load_config, generate_input, predict_loop


def _get_external_parameters():
    parser = argparse.ArgumentParser(description='Train autoencoder model for sequence prediction')
    parser.add_argument('-config', type=str, dest='config_file', metavar='<config file>',
                        required=False, help='configuration file with dicts and max_len', default=None)
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


def create_autoencoder_dataset(words, sentences):
    word2ind = {word: index for index, word in enumerate(words, start=2)}
    ind2word = {str(index): word for index, word in enumerate(words, start=2)}
    sentences_int = [[word2ind[w.lower()] for w in s] for s in sentences]
    word2ind["<PAD>"] = 0
    ind2word[0] = "<PAD>"
    word2ind["<END>"] = 1
    ind2word[1] = "<END>"
    result_list = []
    for sentence in sentences_int:
        result_list.append([sentence, sentence[1:]+[1]])
    return result_list, word2ind, ind2word


def _train_mode():
    def _run(args):
        generator, dataset_len, max_len, max_features, batch_size = generate_input(args, create_autoencoder_dataset)
        nn = AutoEncoder(max_features, args.embedding_size, args.hidden_size, max_len, args.output_directory)
        nn.train_model(generator, batch_size, args.epochs, dataset_len)
        nn.save_model()

    return _run


def _test_mode():
    def _run(args):
        if args.config_file:
            id2word, word2id, max_len = load_config(args.config_file)
        else:
            raise FileNotFoundError("Config path is none")
        nn = AutoEncoder(len(id2word), args.embedding_size, args.hidden_size, max_len)
        nn.model.load_weights(args.model_path)
        predict_loop(PredictModel(nn, (id2word, word2id, max_len)))
    return _run


def main():
    args = _get_external_parameters()
    run_mode = args.command()
    run_mode(args)


if __name__ == '__main__':
    main()
