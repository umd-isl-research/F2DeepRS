import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
import random
import warnings
import config
from utils import load_model

class F2deepRS(nn.Module):
    def __init__(self, num_user=config.num_user, num_item=config.num_item, vec_dim=config.vec_dim):
        super(F2deepRS, self).__init__()
        self.user_embeddings = nn.Embedding(num_user, vec_dim)  # load the pretrained user/item vectors here
        self.item_embeddings = nn.Embedding(num_item, vec_dim)  # load the pretrained user/item vectors here
        self.layers = nn.Sequential(
            nn.Linear(in_features=vec_dim*2, out_features=64),
            nn.LeakyReLU(),
            #nn.Dropout(0.1),
            #nn.SELU(),
            nn.Linear(in_features=64, out_features=64),
            nn.LeakyReLU(),
            #nn.Dropout(0.1),
            #nn.SELU(),
            nn.Linear(in_features=64, out_features=32),
            nn.LeakyReLU(),
            #nn.Dropout(0.1),
            #nn.SELU(),
            nn.Linear(in_features=32, out_features=16),
            nn.LeakyReLU(),
            #nn.Dropout(0.1),
            #nn.SELU(),
            nn.Linear(in_features=16, out_features=1),
            # nn.Sigmoid()  # do not use because nn.BCEWithLogitsLoss applies the sigmoid function
        )

    def forward(self, *input):
        """
        :param input: [uid, iid]
        :param kwargs:
        :return:
        """
        user_embeddings = self.user_embeddings(input[0])
        item_embeddings = self.item_embeddings(input[1])
        vector = torch.cat((user_embeddings, item_embeddings), dim=1)  # do not concatenate along the batch dimension
        # vector = torch.mul(user_embeddings, item_embeddings)  # results show this is not as good as concatenate operation
        return self.layers(vector)

    def init_embeddings(self):
        if os.path.exists(config.user_item_vecs):
            ans = np.load(config.user_item_vecs)
            user_vectors, item_vectors = torch.from_numpy(ans["user_vecs"]), torch.from_numpy(ans["item_vecs"])
            # nn.init.constant_(self.user_embeddings.weight, user_vectors)
            # nn.init.constant_(self.item_embeddings.weight, item_vectors)
            self.user_embeddings.weight.data.copy_(user_vectors)
            self.item_embeddings.weight.data.copy_(item_vectors)
        else:
            warnings.warn("no initial user or item vector found")


class ItemContentEmb(nn.Module):  # use embedding to establish item content vector
    pass

class ItemContentSimNet(nn.Module):
    def __init__(self,input_dim=config.num_genre_type, content_dim=config.content_vec_dim):
        super(ItemContentSimNet,self).__init__()
        self.Encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=512),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=256, out_features=128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=128, out_features=64),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=64, out_features=content_dim)
        )
        self.Out = nn.Linear(in_features=config.content_vec_dim, out_features=1)

    def forward(self, *input):  # for two genre vector, output if they are from the same tree
        input = [F.one_hot(i, num_classes = config.num_genre_type).float() for i in input]
        vec1,vec2 = self.Encoder(input[0]), self.Encoder(input[1])
        #return self.Out(torch.cat((vec1,vec2),1)) # results indicate it is not good to concatenate them
        return self.Out(torch.mul(vec1, vec2))


class F2deepRSwithContent(nn.Module):
    def __init__(self, num_user=config.num_user, num_item=config.num_item, vec_dim=config.vec_dim, content_dim=config.content_vec_dim):
        super(F2deepRSwithContent, self).__init__()
        self.user_embeddings = nn.Embedding(num_user, vec_dim)
        self.item_embeddings = nn.Embedding(num_item, vec_dim)
        itemContentSimNet = ItemContentSimNet()
        self.ItemContentEncoder = itemContentSimNet.Encoder
        self.ItemFusion = nn.Sequential(
            nn.Linear(in_features=vec_dim+content_dim, out_features=vec_dim),
            nn.LeakyReLU()
        )
        self.layers = nn.Sequential(
            nn.Linear(in_features=vec_dim*2, out_features=64),
            nn.LeakyReLU(),
            #nn.Dropout(0.1),
            #nn.SELU(),
            nn.Linear(in_features=64, out_features=64),
            nn.LeakyReLU(),
            #nn.Dropout(0.1),
            #nn.SELU(),
            nn.Linear(in_features=64, out_features=32),
            nn.LeakyReLU(),
            #nn.Dropout(0.1),
            #nn.SELU(),
            nn.Linear(in_features=32, out_features=16),
            nn.LeakyReLU(),
            #nn.Dropout(0.1),
            #nn.SELU(),
            nn.Linear(in_features=16, out_features=1),
            # nn.Sigmoid()  # do not use because nn.BCEWithLogitsLoss applies the sigmoid function
        )

    def forward(self, *input):
        """
        :param input: [uid, iid, genre_id]
        :param kwargs:
        :return:
        """
        user_embeddings = self.user_embeddings(input[0])
        item_embeddings = self.item_embeddings(input[1])
        genre_onehot = F.one_hot(input[2], num_classes = config.num_genre_type).float()
        genre_vector = self.ItemContentEncoder(genre_onehot)
        item_fused = self.ItemFusion(torch.cat((item_embeddings, genre_vector), dim=1))
        vector = torch.cat((user_embeddings, item_fused), dim=1)  # do not concatenate along the batch dimension
        # vector = torch.mul(user_embeddings, item_embeddings)  # results show this is not as good as concatenate operation
        return self.layers(vector)

    def init_embeddings(self):
        if os.path.exists(config.user_item_vecs):
            ans = np.load(config.user_item_vecs)
            user_vectors, item_vectors = torch.from_numpy(ans["user_vecs"]), torch.from_numpy(ans["item_vecs"])
            # nn.init.constant_(self.user_embeddings.weight, user_vectors)
            # nn.init.constant_(self.item_embeddings.weight, item_vectors)
            self.user_embeddings.weight.data.copy_(user_vectors)
            self.item_embeddings.weight.data.copy_(item_vectors)
        else:
            warnings.warn("no initial user or item vector found")
    def init_genre_encoder(self):
        if os.path.exists(os.path.join(config.model_root, config.genre_encoder)):
            self.ItemContentEncoder = load_model(self.ItemContentEncoder, os.path.join(config.model_root, config.genre_encoder))
        else:
            warnings.warn("no genre encoder found")

if __name__=="__main__":
    instance = F2deepRS(200000,136736,64)
    #instance = F2DeepRSwithContent(200000, 136736, 64,64)

