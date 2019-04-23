from preprocess import preprocess

UNK_INDEX = 0


class Vocab:
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


def Lang(vocabs, sent_in):
    statistic = {"sent_num": 0, "word_num": 0, "vocab_size": 0, "vocab_size_>3": 0}
    # Build Vocab
    for sent in sent_in:
        vocabs.index_words(sent)
    # Statistic
    # 1. Number of sentences
    statistic["sent_num"] = len(sent_in)

    # 2. Number of words
    statistic['word_num'] = vocabs.word_num

    # 3. Number of unique words
    statistic['vocab_size'] = vocabs.n_words

    # 4. Number of unique words && > 3
    for vocab in vocabs.word2index:
        if vocab is not "UNK" and vocab is not "<Start>" and vocab is not "<End>":
            if vocabs.word2count[vocab] > 3:
                statistic['vocab_size_>3'] += 1

    # 5. Most Frequent Words
    statistic['frequent_word'] = sorted(vocabs.word2count.items(), key=
    lambda kv: (kv[1], kv[0]), reverse=True)[0:10]

    return statistic, vocabs


def getVocab():
    vocab = Vocab()
    sent_in, sent_out = preprocess("dataset/micro/train.txt")
    statistic, vocab = Lang(vocab, sent_in)
    print(statistic)
    return vocab, vocab


# Testing
# getVocab()
