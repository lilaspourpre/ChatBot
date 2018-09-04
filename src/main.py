import os
os.environ['KERAS_BACKEND']='tensorflow'
from entities.Seq2Seq import CustomSeq2Seq
from entities.GAN3 import GAN3
from entities.Transformer import Transformer
from dataset_creator import *


def __get_external_parameters():
    parser = argparse.ArgumentParser(description='Train sequence model for dialogues')
    parser.add_argument('-o', type=str, dest='output_directory', metavar='<directory>',
                        required=True, help='directory for results')
    parser.add_argument('-i', type=str, dest='input_file', metavar='<input file>',
                        required=True, help='input file (raw or prepared json)')
    parser.add_argument('-c', type=str, dest='config_file', metavar='<config file>',
                        required=False, help='configuration file with dicts and max_len', default=None)
    parser.add_argument('-m', type=choose_model, dest='model', metavar='<model>',
                        required=False, choices=(CustomSeq2Seq, GAN3), help='model to use: seq2seq or gan or transformer',
                        default="transformer")
    parser.add_argument('-e', type=int, dest='epochs', metavar='<epochs>',
                        required=False, help='number of epochs', default=500)
    parser.add_argument('-b', type=int, dest='batch_size', metavar='<batch size>',
                        required=False, help='batch size', default=16)
    parser.add_argument('-s', type=int, dest='embedding_size', metavar='<embedding size>',
                        required=False, help='embedding size', default=128)
    parser.add_argument('-l', type=int, dest='hidden_size', metavar='<hidden size>',
                        required=False, help='hidden size', default=128)
    args = parser.parse_args()
    directory = args.output_directory
    input_file = args.input_file
    config_file = args.config_file
    model = args.model
    epochs = args.epochs
    batch_size = args.batch_size
    embedding_size = args.embedding_size
    hidden_size = args.hidden_size
    return directory, input_file, config_file, model, epochs, batch_size, embedding_size, hidden_size


def choose_model(encoder_type):
    if encoder_type.lower() == "seq2seq":
        return CustomSeq2Seq
    elif encoder_type.lower() == "gan":
        return GAN3
    elif encoder_type.lower() == "transformer":
        return Transformer
    else:
        raise Exception("Unknown encoder name {}".format(encoder_type))


def read_json_file(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        sentences_list = f.readlines()
    return [json.loads(sentence) for sentence in sentences_list[:-1]]


def load_config(config_file):
    config = json.load(open(config_file, 'r'))
    return config["id2word"], config["word2id"], config["max_len"]


def main():
    target_directory, input_file, config_file, nn_model, epochs, batch_size, embedding_size, hidden_size = \
        __get_external_parameters()
    if config_file:
        data_set = read_json_file(input_file)
        id2word, word2id, max_len = load_config(config_file)
    else:
        sentences = read_txt_file(input_file)
        data_set, id2word, word2ind, max_len = create_dataset(sentences)
        write_to_json_config(os.path.join(target_directory, "config.json"), id2word, word2ind, max_len)
    max_features = len(id2word)
    nn = nn_model(max_features, embedding_size, hidden_size, max_len, target_directory)
    model = nn.train_model(data_set, max_len, max_features, batch_size, epochs)
    model.save(os.path.join(target_directory, "final_model.h5"))


if __name__ == '__main__':
    main()
