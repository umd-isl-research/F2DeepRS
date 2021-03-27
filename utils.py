import os
import torch
import torch.nn as nn
import random
import json
import warnings
import config
import matplotlib.pyplot as plt
import numpy as np

def loadalluserinfo(userinfopath):
    if os.path.exists(userinfopath):
        with open(userinfopath, 'r') as f:
            alluserinfo = json.load(f)
            print("alluserinfo is loaded")
        return alluserinfo
    else:
        warnings.warn("cannot find the file alluserinfo.json, double check")

def make_variable(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

def init_weights(layer):
    layer_name = layer.__class__.__name__
    if layer_name.find("Linear") != -1:  # if isinstance(layer, nn.Linear)
        nn.init.xavier_normal_(layer.weight)  # normal way

        #nn.init.kaiming_normal_(layer.weight, a=1, mode='fan_in')  # particularly for SELU activation using torch built in function

        #fan_in, fan_out = layer.weight.size(1), layer.weight.size(0)
        #layer.weight.data.copy_(torch.normal(0, torch.sqrt(torch.tensor(1./fan_in)), layer.weight.size()))# particularly for SELU activation, called Lecun_norm
        #nn.init.normal_(layer.weight, mean=0, std=torch.sqrt(torch.tensor(1./fan_in)))
        layer.bias.data.fill_(0.01)

    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0.01)  # original 0
    elif layer_name.find("Conv") != -1:
        nn.init.kaiming_normal_(layer.weight)  # layer.weight.data.normal_(0.0, 0.02)
        layer.bias.data.fill_(0.01)
    elif layer_name.find("Embedding")!=-1:
        nn.init.xavier_uniform_(layer.weight)

def load_model(net, w_path):
    net.apply(init_weights)
    if w_path is not None and os.path.exists(w_path):
        net.load_state_dict(torch.load(w_path))
        net.restored = True
        print("Restore model from: {}".format(os.path.abspath(w_path)))
    else:
        print("randomly initialize the model")
    return net

def save_model(net, filename):
    """Save trained model."""
    if not os.path.exists(config.model_root):
        os.makedirs(config.model_root)
    torch.save(net.state_dict(), os.path.join(config.model_root, filename))
    print("save model to: {}".format(os.path.join(config.model_root, filename)))


def net_freeze(net):
    for param in net.parameters():
        param.requires_grad = False
    return net

def net_unfreeze(net):
    for param in net.parameters():
        param.requires_grad = True
    return net

def init_random_seed(manual_seed):
    """Init random seed."""
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
