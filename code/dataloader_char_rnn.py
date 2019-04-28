import pandas as pd
import torch
import torch.utils.data as data
from statistics_char_rnn import getVocab
from preprocess_char_rnn import preprocess

UNK_INDEX = 0


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, sent_in, sent_out, vocab):
        self.X = sent_in
        self.y = sent_out
        self.vocab = vocab
        self.num_total_seqs = len(self.X)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        X = self.tokenize(self.X[index])
        if self.y is not None:
            y = self.tokenize(self.y[index])
            return torch.LongTensor(X), torch.LongTensor(y)
        else:
            return torch.LongTensor(X)

    def __len__(self):
        return self.num_total_seqs

    def tokenize(self, sentence):
        return [self.vocab.char2index[char] for char in sentence]


def get_dataloaders(batch_size, window_size):
    vocab = getVocab()
    train_data_sent_in, train_data_sent_out = preprocess("dataset/micro/train.txt", windows=window_size)
    dev_data_sent_in, dev_data_sent_out = preprocess("dataset/micro/valid.txt", windows=window_size)
    test_data_sent_in, test_data_sent_out = preprocess("dataset/micro/test.txt", windows=window_size)
    train = Dataset(train_data_sent_in, train_data_sent_out, vocab)
    dev = Dataset(dev_data_sent_in, dev_data_sent_out, vocab)
    test = Dataset(test_data_sent_in, test_data_sent_out, vocab)
    data_loader_tr = torch.utils.data.DataLoader(dataset=train,
                                                 batch_size=batch_size,
                                                 shuffle=True)
    data_loader_dev = torch.utils.data.DataLoader(dataset=dev,
                                                  batch_size=batch_size,
                                                  shuffle=False)
    data_loader_test = torch.utils.data.DataLoader(dataset=test,
                                                   batch_size=batch_size,
                                                   shuffle=False)
    return data_loader_tr, data_loader_dev, data_loader_test, len(vocab.index2char), vocab

