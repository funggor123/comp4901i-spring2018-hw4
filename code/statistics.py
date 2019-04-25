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


def Lang(vocabs, sent_in, max_vocab=-1):
    statistic = {"sent_num": 0, "word_num": 0, "vocab_size": 0, "vocab_size_>3": 0}
    # Build Vocab
    for sent in sent_in:
        vocabs.index_words(sent)


    #limit the size of vocab
    if(max_vocab > 0):
        sorted_vocabCount = sorted(vocabs.word2count.items(),reverse=True, key=lambda kv: kv[1])
        unknowWord = 0
        newWord2index = {"UNK": UNK_INDEX}
        newIndex2word = {UNK_INDEX: "UNK"}
        newWord2Count = {}
        index = 1 
        for i , k in sorted_vocabCount:
            if(index < max_vocab):
                newWord2index[i] = index
                newIndex2word[index] = i
                newWord2Count[i] = k
                print(index,i,k)
                index = index+1
            else:
                unknowWord += k
        #print("newWord2index: ",newWord2index, "newIndex2word: ", newIndex2word, "newWord2Count: ", newWord2Count, "unknowWord:", unknowWord, "wordNumber", "newNumWord: ", len(newWord2index) )
        vocabs.word2count = newWord2Count
        vocabs.word2index = newWord2index
        vocabs.index2word = newIndex2word
        vocabs.n_words = len(newWord2index)
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

    # 6. UNK token rate
    statistic['UNK token rate'] = unknowWord/vocabs.word_num
    return statistic, vocabs


def getVocab():
    vocab = Vocab()
    sent_in, sent_out = preprocess("dataset/micro/train.txt")
    # max_vocab = -1 use all vocab
    statistic, vocab = Lang(vocab, sent_in, max_vocab=10000)
    print(statistic)
    return vocab


# Testing
#getVocab()