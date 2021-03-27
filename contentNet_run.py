import numpy as np
import pandas as pd
from sklearn import metrics
import copy
import time
import os
import torch
from torch.optim import Adam
import config
from utils import save_model, load_model

from F2deepRS_net import ItemContentSimNet
from dataset import GenreData, GenreDataset, get_genre_batches
device = config.device
criterion = torch.nn.BCEWithLogitsLoss()

def eval_genre_model(net, testdata):
    e_acc, e_loss, e_f1 = 0, 0, 0
    #num_baches = 0
    for g1,g2,l in get_genre_batches(testdata):
        g1,g2,l = torch.tensor(g1).long().to(device), torch.tensor(g2).long().to(device), torch.tensor(l).unsqueeze(1).float().to(device)
        logits = net(g1,g2)
        loss = criterion(logits, l)
        thresh = torch.FloatTensor([0.5]).to(device)
        probs = torch.sigmoid(logits)
        preds = (probs.data > thresh).float()
        e_acc += (preds == l).sum().item()
        e_loss += loss.item()*g1.size(0)
        e_f1 += metrics.f1_score(y_true=l.squeeze(1).cpu().numpy(), y_pred=preds.cpu().numpy(), pos_label=1, average="binary")*g1.size(0)
        #num_baches += 1
    eval_acc, eval_loss, eval_f1 = e_acc/len(testdata), e_loss/len(testdata), e_f1/len(testdata)
    return eval_acc, eval_loss, eval_f1

if __name__=="__main__":
    genreData = GenreData()
    genre_train = GenreDataset(genreData, istrain=True, split=1.0)
    genre_test = GenreDataset(genreData, istrain=False, split=1.0)
    contentNet = ItemContentSimNet()
    contentNet = load_model(contentNet, os.path.join(config.model_root, "genreSimNet_vec64_0.85train_4000epoch.pt"))  # load existing model, or randomize the weights
    contentNet.to(device)
    best_model = copy.deepcopy(contentNet)
    tr_acc, tr_loss, tr_f1 = [], [], []
    te_acc, te_loss, te_f1 = [], [], []
    contentNet.eval()
    eval_acc, eval_loss, eval_f1 = eval_genre_model(contentNet, genre_train)
    contentNet.train()
    te_acc.append(eval_acc)
    te_loss.append(eval_loss)
    te_f1.append(eval_f1)
    print("Evaluation: acc={:.6f}, loss={:.6f}, f1={:.6f}".format(eval_acc, eval_loss, eval_f1))

    opt = Adam(contentNet.parameters(), lr=config.genre_lr, betas=(config.beta1, config.beta2), weight_decay=1e-4)
    for e in range(config.genreNet_epoch):
        e_acc, e_loss, e_f1 = 0, 0, 0
        for g1,g2,l in get_genre_batches(genre_train):
            g1,g2,l = torch.tensor(g1).long().to(device), torch.tensor(g2).long().to(device), torch.tensor(l).unsqueeze(1).float().to(device)
            opt.zero_grad()
            logits = contentNet(g1,g2)

            loss = criterion(logits, l)
            loss.backward()
            opt.step()

            thresh = torch.FloatTensor([0.5]).to(device)
            preds = (torch.sigmoid(logits).data>thresh).float()
            e_acc += (preds==l).sum().item()
            e_loss += loss.item()*g1.size(0)
            e_f1 += metrics.f1_score(y_true=l.squeeze(1).cpu().numpy(), y_pred=preds.cpu().numpy(), pos_label=1, average="binary")*g1.size(0)
        print('\n')
        print("Training: epoch {} of {}".format(e+1,config.genreNet_epoch),
              "acc={:.6f}, loss={:.6f}, f1={:.6f}".format(e_acc/len(genre_train), e_loss/len(genre_train), e_f1/len(genre_train)))

        if (e+1)%1==0:
            contentNet.eval()
            eval_acc, eval_loss, eval_f1 = eval_genre_model(contentNet, genre_test)
            contentNet.train()
            print("Evaluation: acc={:.6f}, loss={:.6f}, f1={:.6f}".format(eval_acc, eval_loss, eval_f1))

    save_model(contentNet.Encoder, "genreEncoder%d.pt" % time.time())
    save_model(contentNet, "genreSimNet%d.pt" % time.time())