#in this file content only refers to item content
import numpy as np
from sklearn import metrics
import time
import heapq
import copy
import psutil

import torch
from torch import nn
from F2deepRS_net import F2deepRSwithContent
import os
from tqdm import tqdm
import config
from utils import save_model, load_model
from torch.optim import Adam
from dataset import YahooR2Dataset, get_batches, ItemContentData
from letor_metrics import ndcg_score

device = config.device
criterion = nn.BCEWithLogitsLoss()

def eval_model(net, data_test):
    # get recall, neg recall, etc
    e_acc, e_loss, e_f1 = 0, 0, 0
    e_recall, e_neg_recall, e_precision = 0, 0, 0
    num_likdislik = 0
    num_eval_users = len(data_test.testing)
    valid_num_eval_users = 0
    e_ndcg = 0
    num_recommend_items = []
    for u, gt in data_test.testing.items():

        i = np.array(gt["like"]+gt["dislike"]+gt["weak_negs"], dtype=np.int32)
        u = np.full(i.shape[0], u, dtype=np.int32)
        l = np.array([1]*len(gt["like"])+[0]*(len(gt["dislike"])+len(gt["weak_negs"])), dtype=np.float)
        g = item_content[i]
        u,i,g,l = torch.from_numpy(u).long().to(device), torch.from_numpy(i).long().to(device), torch.from_numpy(g).long().to(device), torch.from_numpy(l).unsqueeze(1).float().to(device)
        e_num_likdislik = len(gt["like"])+len(gt["dislike"])
        #u, i, l = u.long().to(device), i.long().to(device), l.unsqueeze(1).float().to(device)
        logits = net(u, i, g)
        loss = criterion(logits, l)
        thresh = torch.FloatTensor([0.5]).to(device)
        probs = torch.sigmoid(logits)
        preds = (probs.data > thresh).float()
        num_likdislik+=e_num_likdislik
        e_acc += (preds == l)[:e_num_likdislik].sum().item()  # it's okoy if e_num_likdislik=0, no problem, accuracy does not take weak_negs into account
        e_loss += loss.item()
        e_f1 += metrics.f1_score(y_true=l.squeeze(1).cpu().numpy()[:num_likdislik], y_pred=preds.cpu().numpy()[:num_likdislik], pos_label=1, average="binary")

        map_item_score = dict(zip(i.tolist(),probs.squeeze(1).tolist()))
        ranklist = heapq.nlargest(config.topN, map_item_score, key=map_item_score.get)
        #ranklist = [i for i,s in map_item_score.items() if s>0.5]
        num_recommend_items.append(len(ranklist))
        e_recall += len(set(ranklist) & set(gt["like"]))/(len(gt["like"])+1e-6)
        e_neg_recall += len(set(ranklist) & set(gt["dislike"]))/(len(gt["dislike"])+1e-6)
        e_precision += len(set(ranklist) & set(gt["like"]))/config.topN
        if len(gt["like"])>0:
            valid_num_eval_users += 1
            e_ndcg += ndcg_score(l.cpu().data.numpy().squeeze(), probs.cpu().data.numpy().squeeze(),k=config.topN)
        #else:
        #    e_ndcg += 0.5

    eval_acc, eval_loss, eval_f1 = e_acc/num_likdislik, e_loss/num_eval_users, e_f1/num_eval_users

    return eval_acc, eval_loss, eval_f1, e_recall/num_eval_users, e_neg_recall/num_eval_users, e_precision/num_eval_users, e_ndcg/(valid_num_eval_users+1e-6), np.mean(num_recommend_items)

if __name__=="__main__":
    data_train = YahooR2Dataset(train=True)
    data_test = YahooR2Dataset(train=False)
    item_content = ItemContentData()
    data_test.get_instance()
    #tr_json_data = load_json_data(config.train_json_file)
    #data_train = YahooR2Dataset_train()
    #data_train.get_instance(tr_json_data)
    deepRSwithContent = F2deepRSwithContent(config.num_user, config.num_item, config.vec_dim, config.content_vec_dim)
    deepRSwithContent = load_model(deepRSwithContent, os.path.join(config.model_root, "None"))  # load existing model, or randomize the weights
    deepRSwithContent.init_embeddings()  # load initial user and item vectors learned by BPR
    deepRSwithContent.init_genre_encoder() # load pre-trained genre_encoder
    deepRSwithContent.to(device)
    best_model = copy.deepcopy(deepRSwithContent)
    deepRSwithContent.eval()
    tr_acc, tr_loss, tr_f1 = [], [], []
    te_acc, te_loss, te_recall, te_neg_recall, te_precision, te_ndcg = [], [], [], [], [], []
    eval_acc, eval_loss, eval_f1, eval_recall, eval_neg_recall, eval_precision, eval_ndcg, recN = eval_model(deepRSwithContent, data_test)
    epoch_recN = []
    epoch_recN.append(recN)
    te_acc.append(eval_acc)
    te_loss.append(eval_loss)
    te_recall.append(eval_recall)
    te_neg_recall.append(eval_neg_recall)
    te_precision.append(eval_precision)
    te_ndcg.append(eval_ndcg)
    print("Evaluation: acc={:.6f}, loss={:.6f}, f1={:.6f}".format(eval_acc, eval_loss, eval_f1),
          "recall={:.6f}, neg_recall={:.6f}, precision={:.6f}, ndcg={:.6f}".format(eval_recall, eval_neg_recall, eval_precision, eval_ndcg))

    opt = Adam([
        {"params":deepRSwithContent.user_embeddings.parameters(), "lr":config.lr_embedding},
        {"params":deepRSwithContent.item_embeddings.parameters(), "lr":config.lr_embedding},
        {"params":deepRSwithContent.ItemContentEncoder.parameters(), "lr":config.lr_genre_encoder},
        {"params":list(deepRSwithContent.ItemFusion.parameters())+list(deepRSwithContent.layers.parameters())}],
        lr=config.lr, betas=(config.beta1, config.beta2), weight_decay=1e-4)
    deepRSwithContent.train()
    best_eval_acc = 0

    #pymem = psutil.Process(os.getpid())
    #pre_mem = pymem.memory_info()[0] / 2 ** 20

    #train_dataloader = get_YahooR2_dataloader(data_train)  # FUCK!! memory leak, and slower than yield
    for e in range(config.epoch):
        data_train.get_instance()
        e_acc, e_loss, e_f1 = 0, 0, 0
        for u,i,l in get_batches(data_train):
            g = item_content[i]
            u,i,g,l = torch.from_numpy(u).long().to(device), torch.from_numpy(i).long().to(device), torch.from_numpy(g).long().to(device), torch.from_numpy(l).unsqueeze(1).float().to(device)
            opt.zero_grad()
            logits = deepRSwithContent(u,i,g)

            loss = criterion(logits, l)
            loss.backward()
            opt.step()

            thresh = torch.FloatTensor([0.5]).to(device)
            preds = (torch.sigmoid(logits).data>thresh).float()
            e_acc += (preds==l).sum().item()
            e_loss += loss.item()*u.size(0)
            e_f1 += metrics.f1_score(y_true=l.squeeze(1).cpu().numpy(), y_pred=preds.cpu().numpy(), pos_label=1, average="binary")*u.size(0)
        #curr_mem = pymem.memory_info()[0] / 2 ** 20
        #print("memory added {:.4f}MB".format(curr_mem - pre_mem))
        #pre_mem = curr_mem

        tr_acc.append(e_acc/len(data_train))
        tr_loss.append(e_loss/len(data_train))
        tr_f1.append(e_f1/len(data_train))
        print('\n')
        print("Training: epoch {} of {}".format(e+1,config.epoch),
              "acc={:.6f}, loss={:.6f}, f1={:.6f}".format(e_acc/len(data_train), e_loss/len(data_train), e_f1/len(data_train))
              )
        if (e+1)%config.verbose==0:
            deepRSwithContent.eval()
            eval_acc, eval_loss, eval_f1, eval_recall, eval_neg_recall, eval_precision, eval_ndcg, recN = eval_model(deepRSwithContent, data_test=data_test)
            epoch_recN.append(recN)
            deepRSwithContent.train()
            te_acc.append(eval_acc)
            te_loss.append(eval_loss)
            te_recall.append(eval_recall)
            te_neg_recall.append(eval_neg_recall)
            te_precision.append(eval_precision)
            te_ndcg.append(eval_ndcg)
            if best_eval_acc<eval_acc:
                best_eval_acc = eval_acc
                best_model = copy.deepcopy(deepRSwithContent)
            print("Evaluation: acc={:.6f}, loss={:.6f}, f1={:.6f}".format(eval_acc, eval_loss, eval_f1),
                  "recall={:.6f}, neg_recall={:.6f}, precision={:.6f}, ndcg={:.6f}".format(eval_recall, eval_neg_recall, eval_precision, eval_ndcg))


    save_model(best_model, "deepRSwithContent%d.pt" % time.time())
    np.savez("training_process.npz", tr_acc=np.array(tr_acc), tr_loss=np.array(tr_loss), tr_f1=np.array(tr_f1),
             te_acc=np.array(te_acc), te_loss=np.array(te_loss), te_recall=np.array(te_recall), te_neg_recall=np.array(te_neg_recall),te_precision=np.array(te_precision), te_ndcg=np.array(te_ndcg))



