import torch.nn as nn
import torch.nn.functional as F


class RNNLM(nn.Module):
    def __init__(self, args, vocab_size, target_size, embedding_matrix=None):
        super(RNNLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, args.embed_dim)
        self.model = nn.RNN(args.embed_dim, args.dim_size, num_layers=args.num_layers, dropout=args.dropout)
        self.linear = nn.Linear(args.dim_size, target_size)


    def forward(self, x):
        ebd = self.embedding(x)
        rnn_out, _ = self.model(ebd)
        out = self.linear(rnn_out)
        return out
