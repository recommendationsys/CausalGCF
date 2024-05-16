"""
Paper: Self-supervised Graph Learning for Recommendation
Author: Jiancan Wu, Xiang Wang, Fuli Feng, Xiangnan He, Liang Chen, Jianxun Lian, and Xing Xie
Reference: https://github.com/wujcan/SGL-Torch
"""

__author__ = "Jiancan Wu"
__email__ = "wujcan@gmail.com"

__all__ = ["CausalGCF"]

import dgl
import networkx as nx
import torch
from torch.autograd import Variable
from torch.serialization import save
import torch.sparse as torch_sp
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

from model.base import AbstractRecommender
from reckit.EmbeddingInitializer import UserEmbeddingML, ItemEmbeddingML, UserEmbeddingDB, ItemEmbeddingDB, \
    UserEmbeddingBC, ItemEmbeddingBC, ItemEmbeddingVG
from util.pytorch import inner_product, l2_loss
from util.pytorch import get_initializer
from util.common import Reduction
from data import PointwiseSamplerV2, PairwiseSamplerV2
import numpy as np
from time import time
from reckit import timer
import scipy.sparse as sp
from util.common import normalize_adj_matrix, ensureDir
from util.pytorch import sp_mat_to_sp_tensor
from reckit import randint_choice


class _LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, norm_adj, n_layers):
        super(_LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        # self.norm_adj = norm_adj
        self.n_layers = n_layers
        self.user_embeddings = nn.Embedding(self.num_users, self.embed_dim)
        self.item_embeddings = nn.Embedding(self.num_items, self.embed_dim)
        self.eps = 2.0
        self.delta_P = Variable(torch.zeros(size=[self.num_users, self.embed_dim]),
                                name='delta_P', requires_grad=False).cuda()  # (users, embedding_size) 全零
        self.delta_Q = Variable(torch.zeros(size=[self.num_items, self.embed_dim]),
                                name='delta_Q', requires_grad=False).cuda()  # (items, embedding_size)
        self.epoch = 0
        self.grad_P_dense = torch.zeros(size=[self.num_users, self.embed_dim]).cuda()
        self.grad_Q_dense = torch.zeros(size=[self.num_items, self.embed_dim]).cuda()
        self.dropout = nn.Dropout(0.1)
        self._user_embeddings_final = None
        self._item_embeddings_final = None
        users_feature = np.loadtxt(
            "../SGL-Torch-main/dataset/ml-1M_polluted/user_emb_ml_1M33_0.1.txt")  # Amazon_Video_Games_polluted/user_emb_VG150_0.2_64.txtDouban_polluted/user_emb_Douban72.txt book-crossing_polluted/user_emb_book_crossing61_0.5.txt Amazon_Video_Games_polluted/user_emb_VG687_0.2_128.txt
        '''items_feature = np.loadtxt(
            "D:/zhx/LightGCN+图增广/SGL-Torch-main-aug_agree_mod_new_feature_sim_/SGL-Torch-main/dataset/Amazon_Video_Games_polluted/VG_feature_fill.txt")
        '''
        items_feature = np.loadtxt(
            "../SGL-Torch-main/dataset/ml-1M_polluted/item_emb_ml_1M33_0.1.txt")  #Amazon_Video_Games_polluted/item_emb_VG150_0.2_64.txt Douban_polluted/item_emb_Douban72.txt book-crossing_polluted/item_emb_book_crossing61_0.5.txt Amazon_Video_Games_polluted/item_emb_VG687_0.2_128.txt
        # self.user_info = torch.Tensor(users_feature).int().detach().cuda()
        self.user_info = torch.Tensor(users_feature).detach().cuda()
        self.item_info = torch.Tensor(items_feature).detach().cuda()
        config = {}
        config['num_publisher'] = 1816
        config['num_location'] = 454
        config['num_author'] = 10803
        config['num_year'] = 65
        config['embedding_dim'] = 64
        config['num_iid'] = self.num_items + 1
        config['num_uid'] = self.num_users + 1
        config['num_rate'] = 6
        config['num_genre'] = 25
        config['num_director'] = 2186
        config['num_year'] = 81
        config['num_gender'] = 2
        config['num_age'] = 7
        config['num_occupation'] = 21
        config['num_zipcode'] = 3402
        config['embedding_dim'] = 16
        self.config = config
        self.embedding_user = self.user_info
        self.embedding_item = self.item_info
        # self.embedding_item = ItemEmbeddingVG(self.config).cuda()
        # self.embedding_user = UserEmbeddingBC(self.config).cuda()
        # self.embedding_item = ItemEmbeddingBC(self.config).cuda()
        #user_feature = self.embedding_user(self.user_info)
        # self.embedding_user = UserEmbeddingDB(self.config).cuda()
        # self.embedding_item = ItemEmbeddingDB(self.config).cuda()
        #self.embedding_user = UserEmbeddingML(self.config).cuda()
        #self.embedding_item = ItemEmbeddingML(self.config).cuda()

        # # weight initialization
        # self.reset_parameters()

    def reset_parameters(self, pretrain=0, init_method="uniform", dir=None):
        if pretrain:
            pretrain_user_embedding = np.load(dir + 'user_embeddings.npy')
            pretrain_item_embedding = np.load(dir + 'item_embeddings.npy')
            pretrain_user_tensor = torch.FloatTensor(pretrain_user_embedding).cuda()
            pretrain_item_tensor = torch.FloatTensor(pretrain_item_embedding).cuda()
            self.user_embeddings = nn.Embedding.from_pretrained(pretrain_user_tensor)
            self.item_embeddings = nn.Embedding.from_pretrained(pretrain_item_tensor)
        else:
            init = get_initializer(init_method)
            init(self.user_embeddings.weight)
            init(self.item_embeddings.weight)

    def forward(self, sub_graph1, sub_graph2, norm_adj, users, items, neg_items):
        user_embeddings, item_embeddings = self._forward_gcn(norm_adj)
        user_embeddings1, item_embeddings1 = self._forward_gcn(sub_graph1)
        user_embeddings2, item_embeddings2 = self._forward_gcn(sub_graph2)

        # Normalize embeddings learnt from sub-graph to construct SSL loss
        user_embeddings1 = F.normalize(user_embeddings1, dim=1)
        item_embeddings1 = F.normalize(item_embeddings1, dim=1)
        user_embeddings2 = F.normalize(user_embeddings2, dim=1)
        item_embeddings2 = F.normalize(item_embeddings2, dim=1)

        user_embs = F.embedding(users, user_embeddings)
        item_embs = F.embedding(items, item_embeddings)
        neg_item_embs = F.embedding(neg_items, item_embeddings)
        user_embs1 = F.embedding(users, user_embeddings1)
        item_embs1 = F.embedding(items, item_embeddings1)
        user_embs2 = F.embedding(users, user_embeddings2)
        item_embs2 = F.embedding(items, item_embeddings2)

        sup_pos_ratings = inner_product(user_embs, item_embs)       # [batch_size]
        sup_neg_ratings = inner_product(user_embs, neg_item_embs)   # [batch_size]
        sup_logits = sup_pos_ratings - sup_neg_ratings              # [batch_size]

        pos_ratings_user = inner_product(user_embs1, user_embs2)    # [batch_size]
        pos_ratings_item = inner_product(item_embs1, item_embs2)    # [batch_size]
        tot_ratings_user = torch.matmul(user_embs1,
                                        torch.transpose(user_embeddings2, 0, 1))  # [batch_size, num_users]
        tot_ratings_item = torch.matmul(item_embs1,
                                        torch.transpose(item_embeddings2, 0, 1))  # [batch_size, num_items]

        ssl_logits_user = tot_ratings_user - pos_ratings_user[:, None]  # [batch_size, num_users]
        ssl_logits_item = tot_ratings_item - pos_ratings_item[:, None]  # [batch_size, num_users]
        '''tot_ratings_user1 = torch.matmul(user_embs1,
                                        torch.transpose(user_embeddings2, 0, 1))        # [batch_size, num_users]
        tot_ratings_user2 = torch.matmul(user_embs1,
                                         torch.transpose(user_embeddings1, 0, 1))
        tot_ratings_item1 = torch.matmul(item_embs1,
                                        torch.transpose(item_embeddings2, 0, 1))        # [batch_size, num_items]
        tot_ratings_item2 = torch.matmul(item_embs1,
                                         torch.transpose(item_embeddings1, 0, 1))

        ssl_logits_user1 = tot_ratings_user1 - pos_ratings_user[:, None]                  # [batch_size, num_users]
        ssl_logits_item1 = tot_ratings_item1 - pos_ratings_item[:, None]                  # [batch_size, num_users]
        ssl_logits_user2 = tot_ratings_user2 - pos_ratings_user[:, None]  # [batch_size, num_users]
        ssl_logits_item2 = tot_ratings_item2 - pos_ratings_item[:, None]
        ssl_logits_user = (ssl_logits_user1 + ssl_logits_user2) / 2
        ssl_logits_item = (ssl_logits_item1 + ssl_logits_item2) / 2'''

        return sup_logits, ssl_logits_user, ssl_logits_item

    def _create_adversarial(self, epoch):
        '''self.grad_P, self.grad_Q = torch.autograd.grad(self.loss, [self.embedding_user, self.embedding_item])  # 求梯度

            # convert the IndexedSlice Data to Dense Tensor，将IndexedSlice数据转换为稠密张量
            self.grad_P_dense = torch.gradient(self.grad_P)  # 截断该节点的梯度传播
            self.grad_Q_dense = torch.gradient(self.grad_Q)'''

        # self.grad_P_dense = self.user_embeddings.weight.grad
        # self.grad_Q_dense = self.item_embeddings.weight.
        if epoch > self.epoch:
            self.grad_P_dense = self.user_embeddings.weight.grad
            self.grad_Q_dense = self.item_embeddings.weight.grad
            self.epoch = epoch
        else:
            self.grad_P_dense = self.user_embeddings.weight.grad + self.grad_P_dense
            self.grad_Q_dense = self.item_embeddings.weight.grad + self.grad_Q_dense

        # normalization: new_grad = (grad / |grad|) * eps
        # self.update_P = self.delta_P.assign(torch.nn.functional.normalize(self.grad_P_dense, 1) * self.eps, p=2)
        # self.update_Q = self.delta_Q.assign(torch.nn.functional.normalize(self.grad_Q_dense, 1) * self.eps, p=2)
        self.delta_P = torch.nn.functional.normalize(self.grad_P_dense, p=2, dim=1) * self.eps
        self.delta_Q = torch.nn.functional.normalize(self.grad_Q_dense, p=2, dim=1) * self.eps

    def _forward_gcn(self, norm_adj, adv=False):
        if adv:
            ego_embeddings = torch.cat([self.user_embeddings.weight+self.delta_P, self.item_embeddings.weight+self.delta_Q], dim=0)
        else:
            ego_embeddings = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        all_embeddings = [ego_embeddings]

        for k in range(self.n_layers):
            if isinstance(norm_adj, list):
                ego_embeddings = torch_sp.mm(norm_adj[k], ego_embeddings)
            else:
                ego_embeddings = torch_sp.mm(norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)

        return user_embeddings, item_embeddings

    def predict(self, users):
        if self._user_embeddings_final is None or self._item_embeddings_final is None:
            raise ValueError("Please first switch to 'eval' mode.")
        user_embs = F.embedding(users, self._user_embeddings_final)
        temp_item_embs = self._item_embeddings_final
        ratings = torch.matmul(user_embs, temp_item_embs.T)
        return ratings

    def eval(self, adv, norm_adj):
        super(_LightGCN, self).eval()
        self._user_embeddings_final, self._item_embeddings_final = self._forward_gcn(norm_adj, adv)


class CausalGCF(AbstractRecommender):
    def __init__(self, config):
        super(CausalGCF, self).__init__(config)

        self.sim_i = None
        self.sim_u = None
        self.sim = None
        self.config = config
        self.model_name = config["recommender"]
        self.dataset_name = config["dataset"]

        # General hyper-parameters
        self.reg = config['reg']
        self.emb_size = config['embed_size']
        self.batch_size = config['batch_size']
        self.test_batch_size = config['test_batch_size']
        self.epochs = config["epochs"]
        self.verbose = config["verbose"]
        self.stop_cnt = config["stop_cnt"]
        self.learner = config["learner"]
        self.lr = config['lr']
        self.param_init = config["param_init"]

        # Hyper-parameters for GCN
        self.n_layers = config['n_layers']

        # Hyper-parameters for SSL
        self.ssl_aug_type = config["aug_type"].lower()
        assert self.ssl_aug_type in ['nd','ed', 'rw']
        self.ssl_reg = config["ssl_reg"]
        self.ssl_ratio = config["ssl_ratio"]
        self.ssl_mode = config["ssl_mode"]
        self.ssl_temp = config["ssl_temp"]

        # Other hyper-parameters
        self.best_epoch = 0
        self.best_result = np.zeros([2], dtype=float)

        self.model_str = '#layers=%d-reg=%.0e' % (
            self.n_layers,
            self.reg
        )
        self.model_str += '/ratio=%.1f-mode=%s-temp=%.2f-reg=%.0e' % (
            self.ssl_ratio,
            self.ssl_mode,
            self.ssl_temp,
            self.ssl_reg
        )
        self.pretrain_flag = config["pretrain_flag"]
        if self.pretrain_flag:
            self.epochs = 0
        self.save_flag = config["save_flag"]
        self.save_dir, self.tmp_model_dir = None, None
        if self.pretrain_flag or self.save_flag:
            self.tmp_model_dir = config.data_dir + '%s/model_tmp/%s/%s/' % (
                self.dataset_name, 
                self.model_name,
                self.model_str)
            self.save_dir = config.data_dir + '%s/pretrain-embeddings/%s/n_layers=%d/' % (
                self.dataset_name, 
                self.model_name,
                self.n_layers,)
            ensureDir(self.tmp_model_dir)
            ensureDir(self.save_dir)
        self.adv = False

        self.num_users, self.num_items, self.num_ratings = self.dataset.num_users, self.dataset.num_items, self.dataset.num_train_ratings

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #self.device = torch.device('cpu')
        adj_matrix = self.create_adj_mat()
        self.norm_adj = sp_mat_to_sp_tensor(adj_matrix).to(self.device)


        '''norm_adj = sp.load_npz('D:\zhx\LightGCN+图增广\SGL-Torch-main-aug_agree\SGL-Torch-main\dataset\yelp2018\s_pre_adj_mat_new_u0.05_11.npz')
        self.norm_adj = sp_mat_to_sp_tensor(norm_adj).to(self.device)'''

        self.lightgcn = _LightGCN(self.num_users, self.num_items, self.emb_size,
                                  1, self.n_layers).to(self.device)
        if self.pretrain_flag:
            self.lightgcn.reset_parameters(pretrain=self.pretrain_flag, dir=self.save_dir)
        else:
            self.lightgcn.reset_parameters(init_method=self.param_init)
        self.optimizer = torch.optim.Adam(self.lightgcn.parameters(), lr=self.lr, weight_decay=1e-6)  # 增加权重衰减

        # self.sim = self.get_sim(self.lightgcn.embedding_user(self.lightgcn.user_info),
        #                         self.lightgcn.embedding_item(self.lightgcn.item_info))
        # self.sim_u = self.get_sim_u(self.lightgcn.embedding_user(self.lightgcn.user_info),
        #                             self.lightgcn.embedding_user(self.lightgcn.user_info))
        # self.sim_i = self.get_sim_i(self.lightgcn.embedding_item(self.lightgcn.item_info),
        #                             self.lightgcn.embedding_item(self.lightgcn.item_info))
        self.sim_o = self.get_sim(self.lightgcn.embedding_user, self.lightgcn.embedding_item)
        self.sim_u_o = self.get_sim_u(self.lightgcn.embedding_user, self.lightgcn.embedding_user)
        self.sim_i_o = self.get_sim_i(self.lightgcn.embedding_item, self.lightgcn.embedding_item)

    # 度增强操作
    def get_node_dist(self, graph):
        """
        Compute adjacent node distribution.
        """
        row, col = graph.edges()[0], graph.edges()[1]  # 边的两端节点的索引
        num_node = graph.num_nodes()

        dist_list = []
        for i in range(num_node):
            dist = torch.zeros([num_node], dtype=torch.float32, device='cpu')
            # idx = row[(col == i)]
            idx = col[(row == i)]
            dist[idx] = 1
            dist_list.append(dist)
        dist_list = torch.stack(dist_list, dim=0)
        return dist_list

    def get_sim(self, embeds1, embeds2):
        # normalize embeddings across feature dimension
        '''e1 = torch.spmm(self.tmp_adj, embeds2)
        e2 = torch.spmm(self.tmp_adj.t(), embeds1)
        embeds1 = F.normalize(e1)
        embeds2 = F.normalize(e2)'''
        embeds1 = F.normalize(embeds1)
        embeds2 = F.normalize(embeds2)
        sim = torch.mm(embeds1, embeds2.t()).to('cpu')
        # min = torch.min(sim)
        # u_empty = -1 * torch.Tensor(np.ones(shape=(self.num_items - self.num_users, self.num_items))).to(self.device)
        # sim = torch.cat([sim, u_empty])
        if self.num_items > self.num_users:
            u_empty = -1 * torch.Tensor(
                np.ones(shape=(self.num_items - self.num_users, self.num_items)))
            sim = torch.cat([sim, u_empty])
        else:
            u_empty = -1 * torch.Tensor(
                np.ones(shape=(self.num_users, self.num_users - self.num_items)))
            sim = torch.cat([sim, u_empty], dim=1)
        ones = torch.ones(size=sim.shape)
        sim = (sim + ones) / 2  # 确保相似度范围0~1
        return sim

    def get_sim_u(self, embeds1, embeds2):
        # normalize embeddings across feature dimension
        '''e1 = torch.spmm(self.tmp_adj, embeds1)
        e2 = torch.spmm(self.tmp_adj, embeds2)
        embeds1 = F.normalize(e1)
        embeds2 = F.normalize(e2)'''
        embeds1 = F.normalize(embeds1)
        embeds2 = F.normalize(embeds2)
        sim = torch.mm(embeds1, embeds2.t())
        return sim

    def get_sim_i(self, embeds1, embeds2):
        # normalize embeddings across feature dimension
        '''e1 = torch.spmm(self.tmp_adj.t(), embeds1)
        e2 = torch.spmm(self.tmp_adj.t(), embeds2)
        embeds1 = F.normalize(e1)
        embeds2 = F.normalize(e2)'''
        embeds1 = F.normalize(embeds1)
        embeds2 = F.normalize(embeds2)
        sim = torch.mm(embeds1, embeds2.t())
        return sim

    def neighbor_sampling(self, src_idx, dst_idx, node_dist, sim, sim_u,
                          max_degree, aug_degree):
        phi = sim_u[src_idx, dst_idx].unsqueeze(dim=1).to('cpu')
        #phi = torch.clamp(phi, 0, 0.5)
        phi = torch.clamp(phi, 0, 0.4)

        # print('phi', phi)
        # mix_dist = torch.multiply(node_dist[dst_idx], sim[dst_idx]) * phi+ node_dist[src_idx] * (1 - phi)
        mix_dist = torch.multiply(node_dist[dst_idx], sim[dst_idx]) * phi + node_dist[src_idx] * (1 - phi)

        new_tgt = torch.multinomial(mix_dist + 1e-12, int(max_degree)).to(phi.device)
        #max = torch.max(new_tgt)
        tgt_idx = torch.arange(max_degree).unsqueeze(dim=0).to(phi.device)
        #idx = tgt_idx - aug_degree.unsqueeze(dim=1)

        new_col = new_tgt[(tgt_idx - aug_degree.unsqueeze(dim=1) < 0)]   # 根据度增强个数取出最相近的前几个
        new_row = src_idx.repeat_interleave(aug_degree)
        #print(new_row)
        #print(new_col)
        return new_row, new_col

    def csr_to_Tensor(self, csr):
        Acoo = csr.tocoo()
        Apt = torch.sparse.FloatTensor(torch.LongTensor([Acoo.row.tolist(), Acoo.col.tolist()]),
                                       torch.FloatTensor(Acoo.data.astype(np.float)))
        return Apt

    def degree_mask_edge(self, idx, sim, max_degree, node_degree, mask_prob, node_dist):
        aug_degree = (node_degree * (1 - mask_prob)).long()
        idx = idx.to("cpu")
        sim = sim.to("cpu")
        sim_dist = torch.multiply(node_dist, sim)[idx]
        #sim_dist = sim[idx].to("cpu")

        # _, new_tgt = th.topk(sim_dist + 1e-12, int(max_degree))
        new_tgt = torch.multinomial(sim_dist + 1e-12, int(max_degree))
        tgt_idx = torch.arange(max_degree).unsqueeze(dim=0)

        new_col = new_tgt[(tgt_idx - aug_degree.unsqueeze(dim=1) < 0)]
        new_row = idx.repeat_interleave(aug_degree)
        return new_row, new_col



    # 度增强
    def degree_aug(self, graph, degree, edge_mask_rate_1, threshold):
        max_degree = np.max(degree)
        # min = np.min(degree)  # 8
        node_dist = self.get_node_dist(graph)
        src_idx = torch.LongTensor(np.argwhere(degree < threshold).flatten())  # 尾结点
        rest_idx = torch.LongTensor(np.argwhere(degree >= threshold).flatten())  # 头节点
        rest_node_degree = degree[degree >= threshold]

        sim = self.sim
        sim_u = self.sim_u
        sim = torch.clamp(sim, 0, 1)
        #sim = sim - torch.diag_embed(torch.diag(sim))
        sim_u = torch.clamp(sim_u, 0, 1).to("cpu")
        sim_u = (sim_u - torch.diag_embed(torch.diag(sim_u)))
        src_sim = sim_u[src_idx]
        # dst_idx = th.argmax(src_sim, dim=-1).to(x.device)
        dst_idx = torch.multinomial(src_sim + 1e-12, 1).flatten()

        rest_node_degree = torch.LongTensor(rest_node_degree)
        ''' degree_dist = scatter_add(torch.ones(rest_node_degree.size()), rest_node_degree)
        prob = degree_dist.unsqueeze(dim=0).repeat(src_idx.size(0), 1)
        # aug_degree = th.argmax(prob, dim=-1).to(x.device)
        aug_degree = torch.multinomial(prob, 1).flatten().to('cpu')'''
        aug_degree = degree[degree < threshold]
        #aug_degree = torch.LongTensor(aug_degree)  # 尾结点不增加边也不减边
        aug_degree = torch.LongTensor(aug_degree * (1 + 0.2))  # 只增加20%的边,10%,5%
        node_dist = node_dist.to("cpu")
        new_row_mix_1, new_col_mix_1 = self.neighbor_sampling(src_idx, dst_idx, node_dist, sim, sim_u,
                                                         max_degree, aug_degree)  # 加边

        new_row_rest_1, new_col_rest_1 = self.degree_mask_edge(rest_idx, sim, max_degree, rest_node_degree, edge_mask_rate_1, node_dist)  # 减边
        nsrc1 = torch.cat((new_row_mix_1, new_row_rest_1)).cpu()
        ndst1 = torch.cat((new_col_mix_1, new_col_rest_1)).cpu()
        ng1 = dgl.graph((nsrc1, ndst1), num_nodes=self.num_items if self.num_items > self.num_users else self.num_users)
        ng1 = dgl.to_networkx(ng1)
        return ng1

    # 度增强
    def degree_aug_add(self, graph, degree, edge_mask_rate_1, threshold):
        max_degree = np.max(degree)
        # min = np.min(degree)  # 8
        node_dist = self.get_node_dist(graph)
        src_idx = torch.LongTensor(np.argwhere(degree < threshold).flatten())  # 尾结点
        #rest_idx = torch.LongTensor(np.argwhere(degree >= threshold).flatten())  # 头节点
        #rest_node_degree = degree[degree >= threshold]

        sim = self.sim
        sim_u = self.sim_u
        sim = torch.clamp(sim, 0, 1)
        # sim = sim - torch.diag_embed(torch.diag(sim))
        sim_u = torch.clamp(sim_u, 0, 1).to("cpu")
        sim_u = (sim_u - torch.diag_embed(torch.diag(sim_u)))
        src_sim = sim_u[src_idx]
        # dst_idx = th.argmax(src_sim, dim=-1).to(x.device)
        dst_idx = torch.multinomial(src_sim + 1e-12, 1).flatten()

        #rest_node_degree = torch.LongTensor(rest_node_degree)
        ''' degree_dist = scatter_add(torch.ones(rest_node_degree.size()), rest_node_degree)
        prob = degree_dist.unsqueeze(dim=0).repeat(src_idx.size(0), 1)
        # aug_degree = th.argmax(prob, dim=-1).to(x.device)
        aug_degree = torch.multinomial(prob, 1).flatten().to('cpu')'''
        aug_degree = degree[degree < threshold]
        # aug_degree = torch.LongTensor(aug_degree)  # 尾结点不增加边也不减边
        aug_degree = torch.LongTensor(aug_degree * (1 + 0.1))  # 只增加20%的边,10%,5%
        for i in range(len(aug_degree)):
            if aug_degree[i] > max_degree:
                aug_degree[i] = max_degree
        node_dist = node_dist.to("cpu")
        new_row_mix_1, new_col_mix_1 = self.neighbor_sampling(src_idx, dst_idx, node_dist, sim, sim_u,
                                                              max_degree, aug_degree)  # 加边

        # new_row_rest_1, new_col_rest_1 = self.degree_mask_edge(rest_idx, sim, max_degree, rest_node_degree,
        #                                                        edge_mask_rate_1, node_dist)  # 减边
        #nsrc1 = torch.cat((new_row_mix_1, new_row_rest_1)).cpu()
        #ndst1 = torch.cat((new_col_mix_1, new_col_rest_1)).cpu()
        ng1 = dgl.graph((new_row_mix_1.cpu(), new_col_mix_1.cpu()),
                        num_nodes=self.num_items if self.num_items > self.num_users else self.num_users)
        ng1 = dgl.to_networkx(ng1)
        return ng1

    @timer
    def create_adj_mat(self, is_subgraph=False, aug_type='ed', is_aug_graph=False, mask=0, thershold=0, user_g=True, add=False):
        n_nodes = self.num_users + self.num_items
        users_items = self.dataset.train_data.to_user_item_pairs()
        users_np, items_np = users_items[:, 0], users_items[:, 1]
        '''ratings = np.ones_like(users_np, dtype=np.float32)
        self.tmp_adj = sp.csr_matrix((ratings, (users_np, items_np)),
                                     shape=(self.num_items, self.num_items) if self.num_items > self.num_users
                                     else (self.num_users, self.num_users))
        self.tmp_adj = self.csr_to_Tensor(self.tmp_adj).to("cuda")'''

        if is_subgraph and self.ssl_ratio > 0:
            if aug_type == 'nd':
                drop_user_idx = randint_choice(self.num_users, size=int(self.num_users * self.ssl_ratio), replace=False)  # 随机选择对应数量的用户丢弃
                drop_item_idx = randint_choice(self.num_items, size=int(self.num_items * self.ssl_ratio), replace=False)  # 随机选择对应数量的项目丢弃
                indicator_user = np.ones(self.num_users, dtype=np.float32)
                indicator_item = np.ones(self.num_items, dtype=np.float32)
                indicator_user[drop_user_idx] = 0.
                indicator_item[drop_item_idx] = 0.
                diag_indicator_user = sp.diags(indicator_user)
                diag_indicator_item = sp.diags(indicator_item)
                R = sp.csr_matrix(
                    (np.ones_like(users_np, dtype=np.float32), (users_np, items_np)), 
                    shape=(self.num_users, self.num_items))
                R_prime = diag_indicator_user.dot(R).dot(diag_indicator_item)
                (user_np_keep, item_np_keep) = R_prime.nonzero()
                ratings_keep = R_prime.data
                tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep+self.num_users)), shape=(n_nodes, n_nodes))
            if aug_type in ['ed', 'rw']:
                keep_idx = randint_choice(len(users_np), size=int(len(users_np) * (1 - self.ssl_ratio)), replace=False)  # 随机选择对应数量的交互保留
                user_np = np.array(users_np)[keep_idx]
                item_np = np.array(items_np)[keep_idx]
                ratings = np.ones_like(user_np, dtype=np.float32)
                tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+self.num_users)), shape=(n_nodes, n_nodes))
        else:
            if is_aug_graph:
                ratings = np.ones_like(users_np, dtype=np.float32)
                tmp_adj1 = sp.csr_matrix((ratings, (users_np, items_np)), shape=(self.num_items, self.num_items)
                if self.num_items > self.num_users else (self.num_users, self.num_users))
                if user_g:
                    self.sim = self.sim_o
                    self.sim_u = self.sim_u_o
                    graph = dgl.from_scipy(tmp_adj1)
                    degree = np.array(tmp_adj1.sum(1)).squeeze()[:self.num_users]
                    # self.sim = self.get_sim(self.lightgcn.user_embeddings.weight, self.lightgcn.item_embeddings.weight)
                    # self.sim_u = self.get_sim_u(self.lightgcn.user_embeddings.weight, self.lightgcn.user_embeddings.weight).to("cpu")
                    adj_ = self.degree_aug(graph, degree, edge_mask_rate_1=mask, threshold=thershold)
                    adj1 = nx.to_scipy_sparse_matrix(adj_)
                    if add:
                        degree_add = np.array(adj1.sum(1)).squeeze()[:self.num_users]
                        graph_add = dgl.from_scipy(adj1)
                        # self.sim = self.get_sim(self.lightgcn.user_embeddings.weight, self.lightgcn.item_embeddings.weight)
                        # self.sim_u = self.get_sim_u(self.lightgcn.user_embeddings.weight, self.lightgcn.user_embeddings.weight).to("cpu")
                        adj_add = self.degree_aug_add(graph_add, degree_add, edge_mask_rate_1=mask, threshold=3000)
                        adj1 = nx.to_scipy_sparse_matrix(adj_add)
                else:
                    tmp_adj1 = tmp_adj1.T
                    graph = dgl.from_scipy(tmp_adj1)
                    degree = np.array(tmp_adj1.sum(1)).squeeze()[:self.num_items]  # [:self.num_users]
                    #min = np.min(degree)
                    self.sim = self.sim_o.t()
                    self.sim_u = self.sim_i_o
                    adj1 = self.degree_aug(graph, degree, edge_mask_rate_1=mask, threshold=thershold)
                    adj1 = nx.to_scipy_sparse_matrix(adj1).T
                tmp_adj = sp.dok_matrix((self.num_users + self.num_items, self.num_users + self.num_items),
                                        dtype=np.float32)  # 生成论文中的邻接矩阵
                tmp_adj = tmp_adj.tolil()
                adj1 = adj1[:self.num_users, :self.num_items].tolil()
                tmp_adj[:self.num_users, self.num_users:] = adj1
            else:
                ratings = np.ones_like(users_np, dtype=np.float32)
                tmp_adj = sp.csr_matrix((ratings, (users_np, items_np + self.num_users)), shape=(n_nodes, n_nodes))

        adj_mat = tmp_adj + tmp_adj.T

        # normalize adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        np.seterr(divide='ignore', invalid='ignore')  # 消除被除数为0的警告
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        return adj_matrix

    def train_model(self):
        data_iter = PairwiseSamplerV2(self.dataset.train_data, num_neg=1, batch_size=self.batch_size, shuffle=True)                    
        self.logger.info(self.evaluator.metrics_info())
        stopping_step = 0
        for epoch in range(1, self.epochs + 1):
            total_loss, total_bpr_loss, total_reg_loss = 0.0, 0.0, 0.0
            training_start_time = time()
            if epoch > 10:  # 15 10
                self.sim_o = self.get_sim(self.lightgcn.user_embeddings.weight, self.lightgcn.item_embeddings.weight)
                self.sim_u_o = self.get_sim_u(self.lightgcn.user_embeddings.weight, self.lightgcn.user_embeddings.weight)
                self.sim_i_o = self.get_sim_u(self.lightgcn.item_embeddings.weight, self.lightgcn.item_embeddings.weight)
            '''self.sim = self.get_sim(self.lightgcn.user_embeddings.weight, self.lightgcn.item_embeddings.weight)
            self.sim_u = self.get_sim_u(self.lightgcn.user_embeddings.weight, self.lightgcn.user_embeddings.weight)
            self.sim_i = self.get_sim_u(self.lightgcn.item_embeddings.weight, self.lightgcn.item_embeddings.weight)'''
            sub_graph1 = self.create_adj_mat(is_aug_graph=True, mask=0.3,thershold=16)  # thershold=10 , thershold=20  , thershold=7
            sub_graph1 = sp_mat_to_sp_tensor(sub_graph1).to(self.device)
            sub_graph2 = self.create_adj_mat(is_aug_graph=True, mask=0.3, user_g=False, thershold=18)  # thershold=10 , thershold=7  , thershold=12
            #sub_graph2 = self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)
            sub_graph2 = sp_mat_to_sp_tensor(sub_graph2).to(self.device)
            '''# 生成度增强后的矩阵
            norm_adj = self.create_adj_mat()
            self.norm_adj = sp_mat_to_sp_tensor(norm_adj).to(self.device)'''

            '''if self.ssl_aug_type in ['nd', 'ed']:
                sub_graph1 = self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)
                sub_graph1 = sp_mat_to_sp_tensor(sub_graph1).to(self.device)
                sub_graph2 = self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)
                sub_graph2 = sp_mat_to_sp_tensor(sub_graph2).to(self.device)
            else:
                sub_graph1, sub_graph2 = [], []
                for _ in range(0, self.n_layers):
                    tmp_graph = self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)
                    sub_graph1.append(sp_mat_to_sp_tensor(tmp_graph).to(self.device))
                    tmp_graph = self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)
                    sub_graph2.append(sp_mat_to_sp_tensor(tmp_graph).to(self.device))'''
            self.lightgcn.train()
            '''生成对抗扰动
            if epoch > 1:  # yelp2018数据集 18
                self.adv = True
                for bat_users, bat_pos_items, bat_neg_items in data_iter:
                    bat_users = torch.from_numpy(bat_users).long().to(self.device)
                    bat_pos_items = torch.from_numpy(bat_pos_items).long().to(self.device)
                    bat_neg_items = torch.from_numpy(bat_neg_items).long().to(self.device)
                    sup_logits, ssl_logits_user, ssl_logits_item = self.lightgcn(
                        sub_graph1, sub_graph2, bat_users, bat_pos_items, bat_neg_items)

                    # BPR Loss
                    bpr_loss = -torch.sum(F.logsigmoid(sup_logits))

                    # Reg Loss
                    reg_loss = l2_loss(
                        self.lightgcn.user_embeddings(bat_users),
                        self.lightgcn.item_embeddings(bat_pos_items),
                        self.lightgcn.item_embeddings(bat_neg_items),
                    )

                    # InfoNCE Loss
                    clogits_user = torch.logsumexp(ssl_logits_user / self.ssl_temp, dim=1)
                    clogits_item = torch.logsumexp(ssl_logits_item / self.ssl_temp, dim=1)
                    infonce_loss = torch.sum(clogits_user + clogits_item)

                    loss = bpr_loss + self.ssl_reg * infonce_loss + self.reg * reg_loss
                    total_loss += loss
                    total_bpr_loss += bpr_loss
                    total_reg_loss += self.reg * reg_loss
                loss_t = total_loss/self.num_ratings
                self.optimizer.zero_grad()
                loss_t.backward()
                self.lightgcn._create_adversarial()'''

            for bat_users, bat_pos_items, bat_neg_items in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_pos_items = torch.from_numpy(bat_pos_items).long().to(self.device)
                bat_neg_items = torch.from_numpy(bat_neg_items).long().to(self.device)
                '''sup_logits, ssl_logits_user, ssl_logits_item = self.lightgcn(
                    sub_graph1, sub_graph2, bat_users, bat_pos_items, bat_neg_items)'''
                sup_logits, ssl_logits_user, ssl_logits_item = self.lightgcn(
                    sub_graph1, sub_graph2, self.norm_adj, bat_users, bat_pos_items, bat_neg_items)
                
                # BPR Loss
                bpr_loss = -torch.sum(F.logsigmoid(sup_logits))

                # Reg Loss
                reg_loss = l2_loss(
                    self.lightgcn.user_embeddings(bat_users),
                    self.lightgcn.item_embeddings(bat_pos_items),
                    self.lightgcn.item_embeddings(bat_neg_items),
                )

                # InfoNCE Loss
                clogits_user = torch.logsumexp(ssl_logits_user / self.ssl_temp, dim=1)
                clogits_item = torch.logsumexp(ssl_logits_item / self.ssl_temp, dim=1)
                infonce_loss = torch.sum(clogits_user + clogits_item)
                
                loss = bpr_loss + self.ssl_reg * infonce_loss + self.reg * reg_loss
                #loss = bpr_loss + self.reg * reg_loss
                total_loss += loss
                total_bpr_loss += bpr_loss
                total_reg_loss += self.reg * reg_loss
                self.optimizer.zero_grad()
                loss.backward()
                '''if epoch > 18:  # yelp2018数据集 ED18epoch效果最好,训练epoch后进行对抗攻击得到攻击后的结果
                    self.adv = True
                    self.lightgcn._create_adversarial(epoch)'''
                self.optimizer.step()

            self.logger.info("[iter %d : loss : %.4f = %.4f + %.4f + %.4f, time: %f]" % (
                epoch, 
                total_loss/self.num_ratings,  # 总 loss
                total_bpr_loss / self.num_ratings,  # bpr loss
                (total_loss - total_bpr_loss - total_reg_loss) / self.num_ratings,  # infonce loss
                total_reg_loss / self.num_ratings,  # 正则化项 loss
                time()-training_start_time,))

            if epoch % self.verbose == 0 and epoch > self.config['start_testing_epoch']:
                result, flag = self.evaluate_model()
                self.logger.info("epoch %d:\t%s" % (epoch, result))
                if flag:
                    self.best_epoch = epoch
                    stopping_step = 0
                    self.logger.info("Find a better model.")
                    if self.save_flag:
                        self.logger.info("Save model to file as pretrain.")
                        torch.save(self.lightgcn.state_dict(), self.tmp_model_dir)
                        self.saver.save(self.sess, self.tmp_model_dir)
                else:
                    stopping_step += 1
                    if stopping_step >= self.stop_cnt:
                        self.logger.info("Early stopping is trigger at epoch: {}".format(epoch))
                        break

        self.logger.info("best_result@epoch %d:\n" % self.best_epoch)
        if self.save_flag:
            self.logger.info('Loading from the saved best model during the training process.')
            self.lightgcn.load_state_dict(torch.load(self.tmp_model_dir))
            uebd = self.lightgcn.user_embeddings.weight.cpu().detach().numpy()
            iebd = self.lightgcn.item_embeddings.weight.cpu().detach().numpy()
            np.save(self.save_dir + 'user_embeddings.npy', uebd)
            np.save(self.save_dir + 'item_embeddings.npy', iebd)
            buf, _ = self.evaluate_model()
        elif self.pretrain_flag:
            buf, _ = self.evaluate_model()
        else:
            buf = '\t'.join([("%.4f" % x).ljust(12) for x in self.best_result])
        self.logger.info("\t\t%s" % buf)

    # @timer
    def evaluate_model(self):
        flag = False
        self.lightgcn.eval(self.adv, self.norm_adj)
        current_result, buf = self.evaluator.evaluate(self)
        if self.best_result[1] < current_result[1]:
            self.best_result = current_result
            flag = True
        return buf, flag

    def predict(self, users):
        users = torch.from_numpy(np.asarray(users)).long().to(self.device)
        return self.lightgcn.predict(users).cpu().detach().numpy()
