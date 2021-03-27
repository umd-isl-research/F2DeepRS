import numpy as np
import torch
import os
import time
from F2deepRS_run import eval_model
import config
from utils import load_model
from dataset import YahooR2Dataset
from F2deepRS_net import F2deepRS, F2deepRSwithContent

device = config.device

if __name__=="__main__":
    deepRS = F2deepRS()  # if you want to evaluate the F2deepRS model
    # deepRS = F2deepRSwithContent() # if you want to evaluate the DeepCF with content model
    model = load_model(deepRS, os.path.join(config.model_root, "training_process_50of500_epoch100.pt"))
    model.eval()
    model.to(device)
    data_test = YahooR2Dataset(train=False)
    data_test.get_instance()
    time_start = time.time()
    eval_acc, eval_loss, eval_f1, eval_recall, eval_neg_recall, eval_precision, eval_ndcg, recN = eval_model(model, data_test)
    print("Evaluation: acc={:.6f}, loss={:.6f}, f1={:.6f}".format(eval_acc, eval_loss, eval_f1),
          "recall={:.6f}, neg_recall={:.6f}, precision={:.6f}, ndcg={:.6f}".format(eval_recall, eval_neg_recall, eval_precision, eval_ndcg))
    print("time cost: {:.2f} min".format((time.time()-time_start)/60))