import numpy as np
import scipy.sparse as sp
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchdiffeq import odeint

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType


class LightGODE(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(LightGODE, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.gamma = config['gamma']
        self.train_strategy = config['train_strategy']
        self.train_stage = 'pretrain'
        if self.train_strategy == 'MF':
            self.use_mf = True
        elif self.train_strategy == 'GCN':
            self.use_mf = False
        else:
            self.use_mf = None

        # define layers and loss
        self.t = config['t']
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.encoder = ODEEncoder(self.use_mf, self.n_users, self.n_items, self.embedding_size, self.norm_adj, t = torch.tensor([0, self.t]))

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None
        # parameters initialization
        self.apply(xavier_normal_initialization)

    def ft_init(self):
        self.use_mf = False
        self.train_stage = 'finetuning'
        self.encoder.update(use_mf=self.use_mf)

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

    def get_norm_adj_mat(self):
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    @staticmethod
    def alignment(x, y, alpha=2):
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()

    @staticmethod
    def uniformity(x, t=2):
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def forward(self, user, item):
        if self.train_strategy == 'MF_init' and self.train_stage == 'pretrain':
            self.encoder.update(use_mf = self.training)
        user_e, item_e = self.encoder(user, item)
        return F.normalize(user_e, dim=-1), F.normalize(item_e, dim=-1)
    
    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_e, item_e = self.forward(user, item)
        align = self.alignment(user_e, item_e)
        uniform = self.gamma * (self.uniformity(user_e) + self.uniformity(item_e)) / 2

        return align + uniform

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e = self.encoder.user_embedding(user)
        item_e = self.encoder.item_embedding(item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            if self.train_strategy == 'MF_init':
                self.encoder.update(use_mf=self.training)
            self.restore_user_e, self.restore_item_e = self.encoder.get_all_embeddings()
        user_e = self.restore_user_e[user]
        all_item_e = self.restore_item_e

        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)

class ODEEncoder(nn.Module):
    def __init__(self, use_mf, n_users, n_items, emb_size, norm_adj, t = torch.tensor([0,1]), solver='euler', use_w=False):
        super(ODEEncoder, self).__init__()
        self.use_mf = use_mf
        self.use_w = use_w
        self.n_users = n_users
        self.n_items = n_items
        self.norm_adj = norm_adj
        self.t = t
        self.odefunc1hop = ODEFunc(self.n_users, self.n_items, self.norm_adj,  k_hops=1)
        self.solver = solver

        self.user_embedding = torch.nn.Embedding(n_users, emb_size)
        self.item_embedding = torch.nn.Embedding(n_items, emb_size)

    def update(self, use_mf):
        self.use_mf = use_mf

    def get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    # MF init
    def get_all_embeddings(self):
        all_embeddings = self.get_ego_embeddings()

        # not train -> ode for testing
        if not self.use_mf:
            t = self.t.type_as(all_embeddings)
            self.odefunc1hop.update_e(all_embeddings)
            z1 = odeint(self.odefunc1hop, all_embeddings, t, method=self.solver)[1]
            all_embeddings = z1 

        # trian -> MF for training
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings
    
    # MF init forward
    def forward(self, user_id, item_id):
        user_all_embeddings, item_all_embeddings = self.get_all_embeddings()
        u_embed = user_all_embeddings[user_id]
        i_embed = item_all_embeddings[item_id]
        return u_embed, i_embed
    

class ODEFunc(nn.Module):
    def __init__(self, n_users, n_items, adj, k_hops=1):
        super(ODEFunc, self).__init__()
        self.g =adj
        self.n_users = n_users
        self.n_items = n_items

    def update_e(self, emb):
        self.e = emb

    def forward(self, t, x):
        ax = torch.spmm(self.g, x)
        f = ax + self.e
        return f


