import torch.nn as nn
import torch.nn.functional as F


class RNNLM(nn.Module):
    def __init__(self, args, vocab_size, embedding_matrix=None):
        super(RNNLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, args.embed_dim)
        self.rnn = nn.RNN(args.embed_dim, args.dim_size, args.num_layers, dropout=args.dropout, batch_first=True)
        self.linear = nn.Linear(args.dim_size, vocab_size)

    def forward(self, x):
        ebd = self.embedding(x)
        out, (h,c) = self.rnn(ebd)
        out = self.linear(out)
        out = out.reshape(out.size(0) * out.size(1), out.size(2))
        return out




