import argparse
import nltk
import os
import json
from collections import OrderedDict


def _tokenize_sentence(sentence, language):
    tokens = nltk.tokenize.word_tokenize(sentence, language)
    return tokens


def create_dataset(sentences):
    sentences = [_tokenize_sentence(sentence, "english") for sentence in sentences]
    max_len = max([len(i) for i in sentences])
    words = list(set([word.lower() for sent in sentences for word in sent]))
    word2ind = {word: index for index, word in enumerate(words, start=1)}
    ind2word = {index: word for index, word in enumerate(words, start=1)}
    word2ind["<PAD>"] = 0
    ind2word[0] = "<PAD>"
    sentences_int = [[word2ind[w.lower()] for w in s] for s in sentences]
    dataset = nltk.bigrams(sentences_int)
    return list(dataset), ind2word, word2ind, max_len


def read_txt_file(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        sentences_list = f.readlines()
    return sentences_list


def _write_to_json(path, dataset, id2word, word2id, max_len):
    with open(os.path.join(path, "dataset.json"), "w") as f:
        for sentence in dataset:
            json.dump(sentence, f)
            f.write("\n")
    with open(os.path.join(path, "config.json"), "w") as c:
        config = OrderedDict([("id2word", id2word), ("word2id", word2id), ("max_len", max_len)])
        json.dump(config, c)


def _get_external_parameters():
    parser = argparse.ArgumentParser(description='Train sequence model for dialogues')
    parser.add_argument('-d', type=str, dest='directory', metavar='<directory>',
                        required=True, help='directory for results')
    parser.add_argument('-i', type=str, dest='input_file', metavar='<input file>',
                        required=True, help='input file with train set')
    args = parser.parse_args()
    directory = args.directory
    input_file = args.input_file
    return directory, input_file


def main():
    target_directory, input_file = _get_external_parameters()
    sentences = read_txt_file(input_file)
    dataset, id2word, word2id, max_len = create_dataset(sentences)
    _write_to_json(target_directory, dataset, id2word, word2id, max_len)


if __name__ == '__main__':
    main()
