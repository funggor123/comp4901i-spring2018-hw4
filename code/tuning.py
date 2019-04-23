import torch
import torch.nn as nn

from dataloader import get_dataloaders
from main import trainer
from RNN import RNNLM


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def setup(args, vocab_size, embedding_matrix=None):
    # build model
    # try to use pretrained embedding here
    model = RNNLM(args, vocab_size, embedding_matrix=None)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # choose optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)
    model, best_perp = trainer(train_loader, dev_loader, model, optimizer, criterion, early_stop=args.early_stop)

    return best_perp


lr = 0.0001
dim_size = [128, 256, 512]
num_layers = [1, 2]

args = {
    'lr': lr,
    'dim_size': dim_size[0],
    'num_layers': num_layers[0],
    'window_size': 30,
    'embed_dim': 128,
    'batch_size': 20,
    'dropout': 0.3,
    'early_stop': 3,
}

args = Struct(**args)
print(args)

best_args = args

# load data
train_loader, dev_loader, test_loader, vocab_size, vocab = get_dataloaders(args.batch_size, args.window_size)

best_perp = 0
for size in dim_size:
    temp_args = args
    temp_args.dim_size = size
    print("Current setting: \nHidden Dimension Size: {}\nNum of Hidden Layers: {}".format(temp_args.dim_size, temp_args.num_layers))

    perp = setup(temp_args, vocab_size, embedding_matrix=None)
    if perp < best_perp:
        best_args = temp_args
    print("Best perplexity: {}, Current Perplexity: {}".format(best_perp, perp))
    print("-" * 20)

for layer in num_layers:
    temp_args = args
    temp_args.num_layers = layer
    print("Current setting: \nHidden Dimension Size: {}\nNum of Hidden Layers: {}".format(temp_args.dim_size, temp_args.num_layers))

    perp = setup(args, vocab_size, embedding_matrix=None)
    if perp < best_perp:
        best_args = temp_args
    print("Best perplexity: {}, Current Perplexity: {}".format(best_perp, perp))
    print("-" * 20)

print("Best Perplexity: {}".format(best_perp))
print("Best args: {}".format(best_args))
