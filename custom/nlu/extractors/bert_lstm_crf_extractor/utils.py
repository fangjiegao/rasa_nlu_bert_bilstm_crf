# coding=utf-8
import os, random
import numpy as np
import regex as re
from rasa.shared.nlu.training_data.message import Message
from typing import Any, Dict, Hashable, List, Optional, Set, Text, Tuple, Type, Iterable

MAX_BERT_LEN = 512
BERT_VECTOR_SIZE = 768


def read_dictionary(vocab_path: Text) -> Optional[Dict[Text, int]]:
    word2id = dict()
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        index = 0
        for line in fr:
            if bytes.decode(line).strip() not in word2id.keys():
                word2id[bytes.decode(line).strip()] = index
                index = index + 1
    return word2id


def read_data_for_label(data_dir: Text) -> Optional[Dict[Text, int]]:
    tag = re.compile(r'\((.+?)\)')
    tag2label = dict()
    files_list = os.listdir(data_dir)
    for i in range(0, len(files_list)):
        data_path = os.path.join(data_dir, files_list[i])
        with open(data_path, 'rb') as fr:
            index = 1
            for line in fr:
                res = tag.findall(bytes.decode(line).strip())
                for _ in res:
                    if _ not in tag2label.keys():
                        tag2label[_] = index
                        index = index + 1
    tag2label["O"] = 0
    return tag2label


def pad_labels(labels: List[Text], pad_mark=0):  # "O":0
    """
    :param labels:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x: len(x), labels))
    seq_list, seq_len_list = [], []
    for seq in labels:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def pad_sequences(sequences, labels):
    max_len = max(map(lambda x: len(x), labels))
    features_list = []
    for sequence in sequences:
        temp = np.zeros((max_len - sequence.shape[0], BERT_VECTOR_SIZE))
        features = np.vstack((sequence, temp))
        features_list.append(features)
    return features_list


def pad_sequences_one(sequence, max_len):
    temp = np.zeros((max_len - sequence.shape[0], BERT_VECTOR_SIZE))
    features = np.vstack((sequence, temp))
    return features


def batch_yield(data, batch_size, tag2label, shuffle=False):
    """

    :param data:
    :param batch_size:
    :param tag2label:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data)

    seqs, labels, texts = [], [], []
    for (sent_seqs, tag_, text) in data:
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels, texts
            seqs, labels, texts = [], [], []

        seqs.append(sent_seqs)
        labels.append(label_)
        texts.append(text)

    if len(seqs) != 0:
        yield seqs, labels, texts


def conlleval(label_predict, label_path, metric_path):
    """
    :param label_predict:
    :param label_path:
    :param metric_path:
    :return:
    """
    with open(label_path, "w") as fw:
        line = []
        for sent_result in label_predict:
            for char, tag, tag_ in sent_result:
                char = char.encode("utf-8")
                line.append("{} {} {}\n".format(bytes.decode(char).strip(), tag, tag_))
            line.append("\n")
        fw.writelines(line)
    # with open(metric_path) as fr:
    #     metrics = [line.strip() for line in fr]
    # return metrics


def get_max_len(examples: List[Message]) -> int:
    return max([len(example.data.get('text')) for example in examples])


if __name__ == '__main__':
    word2id_ = read_dictionary('../../../../data/bert-checkpoint/vocab.txt')
    print(word2id_)
    tag2label_ = read_data_for_label('../../../../data/entity/example')
    print(tag2label_)
