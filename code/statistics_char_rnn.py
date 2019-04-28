UNK_INDEX = 0
START_INDEX = 1
END_INDEX = 2


class Vocab:
    def __init__(self):
        alphabet = " abcdefghijklmnopqrstuvwxyz0123456789,!?.;"
        self.char2index = {"<": 0, ">": 1}
        self.index2char = {0: "<", 1: ">"}
        self.default_words = 2
        self.no_vocab = 2
        for i, char in enumerate(alphabet):
            self.char2index[char] = self.default_words + i
            self.index2char[self.default_words + i] = char
            self.no_vocab += 1

def getVocab():
    vocab = Vocab()
    return vocab
