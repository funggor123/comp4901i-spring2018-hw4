import torch
import torch.nn as nn

from dataloader import get_dataloaders
from main import trainer
from RNN import RNNLM


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def start_tuning():
    lr = 0.0001
    dim_size = [128, 256, 512]
    num_layers = [1, 2]

    args = {
        'lr': lr,
        'dim_size': dim_size[0],
        'num_layers': 1,
        'window_size': 30,
        'embed_dim': 128,
        'batch_size': 20,
        'dropout': 0.3,
        'early_stop': 3,
        'amount_of_vocab': 15000,
    }

    args = Struct(**args)
    # print(args)

    best_args = args

    # load data
    train_loader, dev_loader, test_loader, vocab_size, vocab = get_dataloaders(args.batch_size, args.window_size, args.amount_of_vocab)


    best_perp = 0
    for size in dim_size:
        temp_args = args
        temp_args.dim_size = size
        print("Current setting: \nHidden Dimension Size: {}\nNum of Hidden Layers: {}".format(temp_args.dim_size, temp_args.num_layers))

        perp = setup(temp_args, vocab_size, embedding_matrix=None, _train_loader= train_loader, _dev_loader= dev_loader)
        if(best_perp is 0):
            best_perp = perp
            best_args = temp_args
        elif perp < best_perp:
            best_perp = perp
            best_args = temp_args
        print("Best perplexity: {}, Current Perplexity: {}".format(best_perp, perp))
        print("-" * 20)

    for layer in num_layers:
        temp_args = args
        temp_args.num_layers = layer
        print("Current setting: \nHidden Dimension Size: {}\nNum of Hidden Layers: {}".format(temp_args.dim_size, temp_args.num_layers))

        perp = setup(args, vocab_size, embedding_matrix=None, _train_loader= train_loader, _dev_loader= dev_loader)
        if(best_perp is 0):
            best_perp = perp
            best_args = temp_args
        elif perp < best_perp:
            best_perp = perp
            best_args = temp_args
        print("Best perplexity: {}, Current Perplexity: {}".format(best_perp, perp))
        print("-" * 20)

    #-----------------------------------------------
    # train lr = 0.001
    temp_args = args
    temp_args.lr = 0.001
    print("Current setting: \nHidden Dimension Size: {}\nNum of Hidden Layers: {}".format(temp_args.dim_size, temp_args.num_layers))

    perp = setup(args, vocab_size, embedding_matrix=None, _train_loader= train_loader, _dev_loader= dev_loader)
    if(best_perp is 0):
        best_perp = perp
        best_args = temp_args
    elif perp < best_perp:
        best_perp = perp
        best_args = temp_args
    print("Best perplexity: {}, Current Perplexity: {}".format(best_perp, perp))
    print("-" * 20)
    #------------------------------------------------

    print("Best Perplexity: {}".format(best_perp))
    print("Best args: {}".format(best_args))

    print("Use the model with the best Hyper-parameters and report the test set perplexity")
    _best_perp = setup(best_args, vocab_size, _train_loader=train_loader, _dev_loader=test_loader)
    print(_best_perp)


def setup(args, vocab_size, embedding_matrix=None, _train_loader=None, _dev_loader=None):
    # build model
    # try to use pretrained embedding here
    model = RNNLM(args, vocab_size, embedding_matrix=None)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # choose optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)
    model, best_perp = trainer(_train_loader, _dev_loader, model, optimizer, criterion, early_stop=args.early_stop)

    return best_perp

