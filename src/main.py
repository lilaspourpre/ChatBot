import os
os.environ['KERAS_BACKEND']='tensorflow'
from entities.Seq2Seq import CustomSeq2Seq
from entities.GAN import GAN
from dataset_creator import *


def __get_external_parameters():
    #TODO: choose model (custom, seq2seq keras, tf?, gan)
    parser = argparse.ArgumentParser(description='Train sequence model for dialogues')
    parser.add_argument('-o', type=str, dest='output_directory', metavar='<directory>',
                        required=True, help='directory for results')
    parser.add_argument('-i', type=str, dest='input_file', metavar='<input file>',
                        required=True, help='input file (raw or prepared json)')
    parser.add_argument('-c', type=str, dest='config_file', metavar='<config file>',
                        required=False, help='configuration file with dicts and max_len', default=None)
    parser.add_argument('-m', type=choose_model, dest='model', metavar='<model>',
                        required=False, choices=(CustomSeq2Seq, GAN), help='model to use: seq2seq or gan', default="gan")
    args = parser.parse_args()
    directory = args.output_directory
    input_file = args.input_file
    config_file = args.config_file
    model = args.model
    return directory, input_file, config_file, model


def choose_model(encoder_type):
    if encoder_type.lower() == "seq2seq":
        return CustomSeq2Seq
    elif encoder_type.lower() == "gan":
        return GAN
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
    EPOCHS = 50
    BATCH_SIZE = 16

    target_directory, input_file, config_file, nn_model = __get_external_parameters()

    if config_file:
        dataset = read_json_file(input_file)
        id2word, word2id, max_len = load_config(config_file)
    else:
        sentences = read_txt_file(input_file)
        dataset, id2word, word2ind, max_len = create_dataset(sentences)

    embedding_size = 128
    hidden_size = 128
    max_features = len(id2word)
    nn = nn_model(max_features, embedding_size, hidden_size, max_len)
    model = nn.train_model(dataset, max_len, max_features, BATCH_SIZE, EPOCHS)
    model.save(os.path.join(target_directory, "model.h5"))


if __name__ == '__main__':
    main()
