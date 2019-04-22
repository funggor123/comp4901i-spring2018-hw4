import pandas as pd
import re
import numpy as np
import pickle
import unicodedata
import torch
import torch.utils.data as data
import collections
from preprocess import preprocess
UNK_INDEX = 0


class Vocab():
    def __init__(self):
        self.word2index = {"UNK": UNK_INDEX}
        self.word2count = {}
        self.index2word = {UNK_INDEX: "UNK"}
        self.n_words = 1  # Count default tokens
        self.word_num = 0

    def index_words(self, sentence):
        for word in sentence:
            self.word_num += 1
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.word2count[word] = 1
                self.n_words += 1
            else:
                self.word2count[word] += 1

def Lang(vocab, sent_in):
    #doing statistic
    for sent in sent_in:
        vocab.index_words(sent)

    ############################################################
    return vocab


def getVocab():
    vocab = Vocab()
    sent_in, sent_out = preprocess("dataset/micro/train.txt")
    vocab = Lang(vocab, sent_in)
    return vocab
