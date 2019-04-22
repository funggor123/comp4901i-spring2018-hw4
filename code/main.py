import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from dataloader import get_dataloaders
from sklearn.metrics import classification_report, accuracy_score

from RNN import RNNLM

use_gpu = torch.cuda.is_available()


def trainer(train_loader, dev_loader, model, optimizer, criterion, epoch=1000, early_stop=3, scheduler=None):
    best_acc = 0
    for e in range(epoch):
        loss_log = []
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        ######
        if use_gpu:
            model = model.cuda()
        #####
        for i, (seq_in, target) in pbar:

            # use gpu for training
            if use_gpu:
                seq_in = seq_in.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            outputs = model(seq_in)
            loss = criterion(outputs.view(-1, outputs.shape[2]), target.view(-1))
            loss.backward()
            optimizer.step()

            loss_log.append(loss.item())
            pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f}".format((e + 1), np.mean(loss_log)))

        model.eval()
        logits = []
        ys = []
        for seq_in, target in dev_loader:
            ##########
            if use_gpu:
                seq_in = seq_in.cuda()
                target = target.cuda()
            #######
            logit = model(seq_in)
            logits.append(logit.data.cpu().numpy())
            ys.append(target.data.cpu().numpy())
        logits = np.concatenate(logits, axis=0)
        preds = np.argmax(logits, axis=1)
        ys = np.concatenate(ys, axis=0)
        acc = accuracy_score(y_true=ys, y_pred=preds)
        # label_names = ['rating 0', 'rating 1','rating 2']
        # report = classification_report(ys, preds, digits=3,
        #                             target_names=label_names)
        if acc > best_acc:
            best_acc = acc
        else:
            early_stop -= 1
        # print("current validation report")
        # print("\n{}\n".format(report))
        # print()
        print("epcoh: {}, current accuracy:{}, best accuracy:{}".format(e + 1, acc, best_acc))

        if early_stop == 0:
            break
        if scheduler is not None:
            scheduler.step()
    return model, best_acc


def predict(model, test_loader, save_file="submission.csv"):
    logits = []
    inds = []
    model.eval()
    for X, ind in test_loader:
        ###
        if use_gpu:
            X = X.cuda()
        ###
        logit = model(X)
        logits.append(logit.data.cpu().numpy())
        inds.append(ind.data.cpu().numpy())
    logits = np.concatenate(logits, axis=0)
    inds = np.concatenate(inds, axis=0)
    preds = np.argmax(logits, axis=1)
    result = {'id': list(inds), "rating": preds}
    df = pd.DataFrame(result, index=result['id'])
    df.to_csv(save_file)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--embed_dim", type=int, default=150)
    parser.add_argument("--dim_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument("--lr_decay", type=float, default=0.5)
    args = parser.parse_args()

    # load data
    train_loader, dev_loader, test_loader, vocab_size = get_dataloaders(args.batch_size, args.window_size)

    # build model
    # try to use pretrained embedding here
    model = RNNLM(args, vocab_size, target_size=vocab_size, embedding_matrix=None)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # choose optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)
    model, best_acc = trainer(train_loader, dev_loader, model, optimizer, criterion, early_stop=args.early_stop)

    print('best_dev_acc:{}'.format(best_acc))
    predict(model, test_loader)


if __name__ == "__main__":
    main()
