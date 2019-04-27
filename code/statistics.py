from preprocess import clean_str
import operator
from collections import OrderedDict

UNK_INDEX = 0
START_INDEX = 1
END_INDEX = 2


class Vocab:
    def __init__(self):
        self.word2index = {"UNK": UNK_INDEX, "<Start>": START_INDEX, "<End>": END_INDEX}
        self.word2count = {}
        self.index2word = {UNK_INDEX: "UNK", START_INDEX: "<Start>", END_INDEX: "<End>"}
        self.n_words = 3  # Count default tokens
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


def Lang(vocabs, windows, amount_stay=50):
    statistic = {"sent_num": 0, "word_num": 0, "vocab_size": 0, "vocab_size_>3": 0}
    x = []
    y = []
    with open("./dataset/micro/train.txt", "r") as f:
        data = f.read()
        data = clean_str(data)
        data = data.split()
        batch = []
        for word in data:
            if len(batch) == windows - 1:
                vocabs.index_words(batch)
                x += [["<Start>"] + batch]
                y += [batch + ["<End>"]]
                if len(x) > 400:
                    break
                batch = []
            batch += [word]

    # Statistic
    # 1. Number of sentences
    statistic["sent_num"] = len(x)

    # 2. Number of words
    statistic['word_num'] = vocabs.word_num

    # 3. Number of unique words
    statistic['vocab_size'] = vocabs.n_words

    # 4. Number of unique words && > 3
    for vocab in vocabs.word2index:
        if vocab not in ["UNK", "<Start>", "<End>"]:
            if vocabs.word2count[vocab] > 3:
                statistic['vocab_size_>3'] += 1

    # 5. Most Frequent Words
    statistic['frequent_word'] = sorted(vocabs.word2count.items(), key=
    lambda kv: (kv[1], kv[0]), reverse=True)[0:10]

    if amount_stay != -1:
        word2count = set(OrderedDict(sorted(vocabs.word2count.items(), key=lambda kv: kv[1], reverse=True)[:amount_stay]))
        vocabs = Vocab()
        for word in data:
            if len(batch) == windows - 1:
                if word in word2count:
                    vocabs.index_words(batch)


    # 6. UNK token rate
    # statistic['UNK token rate'] = unknowWord / vocabs.word_num
    return statistic, vocabs


def getVocab(window_size):
    vocab = Vocab()
    statistic, vocab = Lang(vocab, window_size, amount_stay=78)
    print(statistic)
    return vocab
