from preprocess import preprocess
import gensim
from gensim.test.utils import datapath

UNK_INDEX = 0


class Vocab:
    def __init__(self):
        # the minimum limit is 10000
        self.word2Vector = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',
                                                                           binary=True, limit=10000)
        print(self.word2Vector)
        self.no_of_vocab = len(self.word2Vector.wv.vectors)
        self.embeddings_matrix = self.word2Vector.wv.vectors


def getVocab():
    vocab = Vocab()
    return vocab

# Testing
# getVocab()
