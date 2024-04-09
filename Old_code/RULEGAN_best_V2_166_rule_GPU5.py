'''
Created on July 1, 2020
PyTorch Implementation of KGIN
@author: Tinglin Huang (tinglin.huang@zju.edu.cn)
'''
__author__ = "huangtinglin"

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_scatter import scatter_mean
import torch_geometric as geometric
import pdb
from torch_geometric.nn import MessagePassing
from torch.nn import Parameter
import pickle
import math
from torch_geometric.utils import remove_self_loops, add_self_loops #, softmax


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, channel, n_hops, n_users,
                 n_factors, n_relations, interact_mat,
                 ind, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_factors = n_factors
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind

        self.temperature = 0.2

        initializer = nn.init.xavier_uniform_
        weight = initializer(torch.empty(n_relations - 1, channel))  # not include interact
        self.weight = nn.Parameter(weight)  # [n_relations - 1, in_channel]

        disen_weight_att = initializer(torch.empty(n_factors, n_relations - 1))
        self.disen_weight_att = nn.Parameter(disen_weight_att)

        for i in range(n_hops):
            self.convs.append(Aggregator(n_users=n_users, n_factors=n_factors))

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    # def _cul_cor_pro(self):
    #     # disen_T: [num_factor, dimension]
    #     disen_T = self.disen_weight_att.t()
    #
    #     # normalized_disen_T: [num_factor, dimension]
    #     normalized_disen_T = disen_T / disen_T.norm(dim=1, keepdim=True)
    #
    #     pos_scores = torch.sum(normalized_disen_T * normalized_disen_T, dim=1)
    #     ttl_scores = torch.sum(torch.mm(disen_T, self.disen_weight_att), dim=1)
    #
    #     pos_scores = torch.exp(pos_scores / self.temperature)
    #     ttl_scores = torch.exp(ttl_scores / self.temperature)
    #
    #     mi_score = - torch.sum(torch.log(pos_scores / ttl_scores))
    #     return mi_score

    def _cul_cor(self):
        def CosineSimilarity(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=0, keepdim=True)
            normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=0, keepdim=True)
            return (normalized_tensor_1 * normalized_tensor_2).sum(dim=0) ** 2  # no negative
        def DistanceCorrelation(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            # ref: https://en.wikipedia.org/wiki/Distance_correlation
            channel = tensor_1.shape[0]
            zeros = torch.zeros(channel, channel).to(tensor_1.device)
            zero = torch.zeros(1).to(tensor_1.device)
            tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
            """cul distance matrix"""
            a_, b_ = torch.matmul(tensor_1, tensor_1.t()) * 2, \
                   torch.matmul(tensor_2, tensor_2.t()) * 2  # [channel, channel]
            tensor_1_square, tensor_2_square = tensor_1 ** 2, tensor_2 ** 2
            a, b = torch.sqrt(torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8), \
                   torch.sqrt(torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8)  # [channel, channel]
            """cul distance correlation"""
            A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
            B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
            dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel ** 2, zero) + 1e-8)
            dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel ** 2, zero) + 1e-8)
            dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel ** 2, zero) + 1e-8)
            return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)
        def MutualInformation():
            # disen_T: [num_factor, dimension]
            disen_T = self.disen_weight_att.t()

            # normalized_disen_T: [num_factor, dimension]
            normalized_disen_T = disen_T / disen_T.norm(dim=1, keepdim=True)

            pos_scores = torch.sum(normalized_disen_T * normalized_disen_T, dim=1)
            ttl_scores = torch.sum(torch.mm(disen_T, self.disen_weight_att), dim=1)

            pos_scores = torch.exp(pos_scores / self.temperature)
            ttl_scores = torch.exp(ttl_scores / self.temperature)

            mi_score = - torch.sum(torch.log(pos_scores / ttl_scores))
            return mi_score

        """cul similarity for each latent factor weight pairs"""
        if self.ind == 'mi':
            return MutualInformation()
        else:
            cor = 0
            for i in range(self.n_factors):
                for j in range(i + 1, self.n_factors):
                    if self.ind == 'distance':
                        cor += DistanceCorrelation(self.disen_weight_att[i], self.disen_weight_att[j])
                    else:
                        cor += CosineSimilarity(self.disen_weight_att[i], self.disen_weight_att[j])
        return cor

    def forward(self, user_emb, entity_emb, latent_emb, edge_index, edge_type,
                interact_mat, mess_dropout=True, node_dropout=False):

        """node dropout"""
        if node_dropout:
            edge_index, edge_type = self._edge_sampling(edge_index, edge_type, self.node_dropout_rate)
            interact_mat = self._sparse_dropout(interact_mat, self.node_dropout_rate)

        entity_res_emb = entity_emb  # [n_entity, channel]
        user_res_emb = user_emb  # [n_users, channel]
        cor = self._cul_cor()
        for i in range(len(self.convs)):
            entity_emb, user_emb = self.convs[i](entity_emb, user_emb, latent_emb,
                                                 edge_index, edge_type, interact_mat,
                                                 self.weight, self.disen_weight_att)

            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)

        return entity_res_emb, user_res_emb, cor

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

def softmax(src, index, num_nodes):
    """
    Given a value tensor: `src`, this function first groups the values along the first dimension
    based on the indices specified in: `index`, and then proceeds to compute the softmax individually for each group.
    """
    #print('src', src)
    #print('index', index)
    #print('num_nodes', num_nodes)
    '''
    N = int(index.max()) + 1 if num_nodes is None else num_nodes
    #print('N', N)
    #print(f"{scatter(src, index, dim=0, dim_size=N, reduce='max')}")
    #print(f"{scatter(src, index, dim=0, dim_size=N, reduce='max')[index]}")
    out = src - scatter(src, index, dim=0, dim_size=N, reduce='max')[index]
    #print('out', out)
    out = out.exp()
    #print('out', out)
    out_sum = scatter(out, index, dim=0, dim_size=N, reduce='sum')[index]
    #print('out_sum', out_sum)
    #print(f'return: {out / (out_sum + 1e-16)}')
    return out / (out_sum + 1e-16)
    '''
    N = int(index.max()) + 1 if num_nodes is None else num_nodes
    out = src - scatter(src, index, dim=0, dim_size=N, reduce='max')[index]
    out = out.exp()
    out_sum = scatter(out, index, dim=0, dim_size=N, reduce='sum')[index]
    return out / (out_sum + 1e-16)


class GATConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1,
                 negative_slope=0.2,
                 dropout=0.,
                 node_dim=-3,
                 bias=True):
        #super(GATConv, self).__init__()
        super(GATConv, self).__init__(node_dim=0, aggr='add')  # "Add" aggregation.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(torch.Tensor(in_channels, heads * out_channels))  # \theta
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))  # \alpha: rather than separate into two parts

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, size=None):
        # 1. Linearly transform node feature matrix (XΘ)
        x = torch.mm(x, self.weight).view(-1, self.heads, self.out_channels)   # N x H x emb(out)
        #print('x', x)

        # 2. Add self-loops to the adjacency matrix (A' = A + I)
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)  # 2 x E
            #print('edge_index', edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))  # 2 x (E+N)
            #print('edge_index', edge_index)

        # 3. Start propagating messages
        return self.propagate(edge_index, x=x, size=size)  # 2 x (E+N), N x H x emb(out), None

    def message(self, x_i, x_j, size_i, edge_index_i):  # Compute normalization (concatenate + softmax)
        # x_i, x_j: after linear x and expand edge (N+E) x H x emb(out)
        # = N x H x emb(in) @ emb(in) x emb(out) (+) E x H x emb(out)
        # edge_index_i: the col part of index
        # size_i: number of nodes
        #print('x_i', x_i)
        #print('x_j', x_j)
        #print('size_i', size_i)
        #print('edge_index_i', edge_index_i)

        x_i = x_i.view(-1, self.heads, self.out_channels)
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)  # (E+N) x H x (emb(out)+ emb(out))
        #print('alpha', alpha)
        alpha = F.leaky_relu(alpha, self.negative_slope)  # LeakReLU only changes those negative.
        #print('alpha', alpha)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)  # Computes a sparsely evaluated softmax
        #print('alpha', alpha)

        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        #print(f'x_j*alpha {x_j * alpha.view(-1, self.heads, 1)}')

        return x_j * alpha.view(-1, self.heads, 1)
        # each row is norm(embedding) vector for each edge_index pair (detail in the following)

    def update(self, aggr_out):  # 4. Return node embeddings (average heads)
        # Based on the directed graph, Node 0 gets message from three edges and one self_loop
        # for Node 1, 2, 3: since they do not get any message from others, so only self_loop

        #print('aggr_out', aggr_out)  # (E+N) x H x emb(out)
        aggr_out = aggr_out.mean(dim=1)  # to average multi-head
        #print('aggr_out', aggr_out)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out


class GraphConv_KGAT(nn.Module):
    """
    Graph Convolutional Network
    embed CKG and using its embedding to calculate prediction score
    """

    def __init__(self, in_channel, out_channel):
        super(GraphConv_KGAT, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv1 = geometric.nn.GATConv(in_channel, out_channel)
        self.conv1 = GATConv(in_channel, out_channel)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, edge_indices):
        #pdb.set_trace()
        x = self.conv1(x, edge_indices)

        x = self.dropout(x)
        x = F.normalize(x)
        return x


class KGAT(nn.Module):
    def __init__(self, data_config, args_config,triplets):
        super(KGAT, self).__init__()
        self.args_config = args_config
        self.data_config = data_config
        self.n_users = data_config["n_users"]
        self.n_items = data_config["n_items"]
        self.n_nodes = data_config["n_nodes"]

        """set input and output channel manually"""
        input_channel = 64
        output_channel = 64
        self.gcn = GraphConv_KGAT(input_channel, output_channel)
        self.triplets = triplets
        self.img_dim_change = nn.Linear(512,64)
        self.txt_dim_change = nn.Linear(384,64)
        self.emb_size = args_config.dim
        self.regs = args_config.l2
        #self.regs = eval(args_config.regs)

        self.all_embed = self._init_weight(triplets)

    def _init_weight(self, triplets):
        all_embed = nn.Parameter(
            torch.FloatTensor(self.n_nodes, self.emb_size), requires_grad=True
        )
        ui = self.n_users + self.n_items

        if self.args_config.pretrain_r:
            nn.init.xavier_uniform_(all_embed)
            all_embed.data[:ui] = self.data_config["all_embed"]
        else:
            nn.init.xavier_uniform_(all_embed)



        img_embedding = pickle.load(open(self.args_config.img_embedding_path, 'rb'))
        txt_embedding = pickle.load(open(self.args_config.txt_embedding_path, 'rb'))
        img_embedding[1] = self.img_dim_change(img_embedding[1])
        txt_embedding[1] = self.txt_dim_change(txt_embedding[1])
        img_embedding[0] = [i + self.n_users for i in img_embedding[0]]
        txt_embedding[0] = [i + self.n_users for i in txt_embedding[0]]
        # img_embedding[1].requires_grad = True
        # txt_embedding[1].requires_grad = True

        comprise_dict = {}
        for i in triplets:
            if i[1] == 1:
                outfit = i[0]
                item = i[2]
                if outfit not in comprise_dict:
                    comprise_dict[outfit] = [item]
                else:
                    comprise_dict[outfit].append(item)

        if self.args_config.item:
            for num, i in enumerate(img_embedding[0]):
                with torch.no_grad():
                    all_embed[i] = 0.5 * (img_embedding[1][num] + txt_embedding[1][num])

        if self.args_config.outfit:
            for outfit in comprise_dict:
                items = comprise_dict[outfit]
                # length = len(items)
                items_embedding = []
                for i in items:
                    index = img_embedding[0].index(i)
                    items_embedding.append(0.5 * (img_embedding[1][index] + txt_embedding[1][index]))
                items_embedding = torch.stack(items_embedding)
                with torch.no_grad():
                    all_embed[outfit] = torch.mean(items_embedding, dim=0)

        if self.args_config.outfit_item:
            for num, i in enumerate(img_embedding[0]):
                with torch.no_grad():
                    all_embed[i] = 0.5 * (img_embedding[1][num] + txt_embedding[1][num])

            for outfit in comprise_dict:
                items = comprise_dict[outfit]
                items_embedding = []
                for i in items:
                    index = img_embedding[0].index(i)
                    items_embedding.append(0.5 * (img_embedding[1][index] + txt_embedding[1][index]))
                items_embedding = torch.stack(items_embedding)
                with torch.no_grad():
                    all_embed[outfit] = torch.mean(items_embedding, dim=0)

        return all_embed

    def build_edge(self, adj_matrix):
        """build edges based on adj_matrix"""
        sample_edge = self.args_config.edge_threshold
        edge_matrix = adj_matrix

        n_node = edge_matrix.size(0)
        node_index = (
            torch.arange(n_node, device=edge_matrix.device)
            .unsqueeze(1)
            .repeat(1, sample_edge)
            .flatten()
        )
        neighbor_index = edge_matrix.flatten()
        edges = torch.cat((node_index.unsqueeze(1), neighbor_index.unsqueeze(1)), dim=1)
        return edges

    def forward(self, user, pos_item, neg_item, edges_matrix, pos_rules, selected_neg_outfits_rules, flag_debug,epoch,flag_user):
        u_e, pos_e, neg_e, pos_rule_e, neg_rule_e= (
            self.all_embed[user],
            self.all_embed[pos_item],
            self.all_embed[neg_item],
            self.all_embed[pos_rules],
            self.all_embed[selected_neg_outfits_rules],
        )

        edges = self.build_edge(edges_matrix)
        x = self.all_embed
        #182423
        gcn_embedding = self.gcn(x, edges.t().contiguous())

        u_e_, pos_e_, neg_e_, pos_rule_e_, neg_rule_e_ = (
            gcn_embedding[user],
            gcn_embedding[pos_item],
            gcn_embedding[neg_item],
            gcn_embedding[pos_rules],
            gcn_embedding[selected_neg_outfits_rules],
        )

        u_e = torch.cat([u_e, u_e_], dim=1)
        pos_e = torch.cat([pos_e, pos_e_], dim=1)
        pos_rule_e = torch.cat([pos_rule_e, pos_rule_e_], dim=1)
        #neg_e = torch.cat([neg_e, neg_e_], dim=1)
        pos_scores = torch.sum(u_e * pos_e, dim=1)
        pos_rule_scores = torch.sum(u_e * pos_rule_e, dim=1)
        if self.args_config.score:
            if self.args_config.score_change:
                if epoch<60:
                    pos_scores += self.args_config.rulescore_pos*pos_rule_scores
                else:
                    pos_scores += self.args_config.rulescore_pos * pos_rule_scores*10
            else:
                pos_scores += self.args_config.rulescore_pos * pos_rule_scores

        '''
        if flag_debug == True:
            pdb.set_trace()
        '''
        if flag_user==False:
            neg_e = torch.cat([neg_e, neg_e_], dim=1)
            neg_rule_e = torch.cat([neg_rule_e, neg_rule_e_], dim=1)
            reg_loss = self._l2_loss(u_e) + self._l2_loss(pos_e) + self._l2_loss(neg_e) + 0.001*self._l2_loss(pos_rule_e) + 0.001*self._l2_loss(neg_rule_e)
            neg_scores = torch.sum(u_e * neg_e, dim=1)
            neg_rule_scores = torch.sum(u_e * neg_rule_e, dim=1)
            if self.args_config.score:
                if self.args_config.score_change:
                    if epoch<60:
                        neg_scores += self.args_config.rulescore_neg*neg_rule_scores
                    else:
                        neg_scores += self.args_config.rulescore_neg*neg_rule_scores*0.1
                else:
                    neg_scores += self.args_config.rulescore_neg * neg_rule_scores


        else: 
            neg_user_e = torch.cat([neg_e, neg_e_], dim=1)
            reg_loss = self._l2_loss(u_e) + self._l2_loss(pos_e) + self._l2_loss(neg_user_e) 
            neg_scores = torch.sum(neg_user_e * pos_e, dim=1)
            neg_rule_scores = torch.sum(neg_user_e * pos_rule_e, dim=1)
            if self.args_config.score:
                if self.args_config.score_change:
                    if epoch<60:
                        neg_scores += self.args_config.rulescore_neg*neg_rule_scores
                    else:
                        neg_scores += self.args_config.rulescore_neg*neg_rule_scores*0.1
                else:
                    neg_scores += self.args_config.rulescore_neg * neg_rule_scores



            #neg_rule_scores = torch.sum(neg_user_e * pos_rule_e, dim=1)
            #neg_scores += neg_rule_scores

        #neg_scores = torch.sum(u_e * neg_e, dim=1)

        # Defining objective function that contains:
        # ... (1) bpr loss
        bpr_loss = torch.log(torch.sigmoid(pos_scores - neg_scores))
        bpr_loss = -torch.mean(bpr_loss)

        # ... (2) emb loss
        reg_loss = self._l2_loss(u_e) + self._l2_loss(pos_e) + self._l2_loss(neg_e)
        reg_loss = self.regs * reg_loss
        return bpr_loss, reg_loss
        #loss = bpr_loss + reg_loss
        #return loss, bpr_loss, reg_loss

    def get_reward(self, users, pos_items, neg_items):
        u_e = self.all_embed[users]
        pos_e = self.all_embed[pos_items]
        neg_e = self.all_embed[neg_items]

        neg_scores = torch.sum(u_e * neg_e, dim=-1)
        ij = torch.sum(neg_e * pos_e, dim=-1)
        neg_scores_2 = - torch.sum(torch.sigmoid(-1e3*u_e * neg_e),dim=-1)
        #reward = 1e-6*neg_scores_2
        #reward = neg_scores + ij + 1e-3*neg_scores_2
        reward = 1e+4*(neg_scores + ij) + 1e-1*neg_scores_2
        return reward

    def get_samples(self, users, neg_users_rand, neg_users_RL, pos_items, neg_items_rand, neg_items_RL, pos_rules, neg_rules, selected_neg_outfits_rules):
        u_e = self.all_embed[users]
        pos_e = self.all_embed[pos_items]
        neg_rand_e = self.all_embed[neg_items_rand]
        neg_RL_e = self.all_embed[neg_items_RL]
        neg_rules_e = self.all_embed[neg_rules]
        neg_RL_rule_e = self.all_embed[selected_neg_outfits_rules]
        neg_rand_u = self.all_embed[neg_users_rand]
        neg_RL_u = self.all_embed[neg_users_RL]
        #outfit
        a = u_e * neg_rand_e
        neg_rand_scores = torch.sum(u_e * neg_rand_e, dim=-1)
        neg_RL_scores = torch.sum(u_e * neg_RL_e, dim = -1)
        #rule score
        neg_rand_rule_scores = torch.sum(u_e * neg_rules_e, dim=1)
        neg_RL_rule_scores = torch.sum(u_e * neg_RL_rule_e, dim = -1)
        if self.args_config.score:
            neg_RL_scores += 0.001*neg_RL_rule_scores
            neg_rand_scores += 0.001*neg_rand_rule_scores
        neg_rand_indices = torch.lt((neg_RL_scores - neg_rand_scores),0)

        # print(neg_rand_indices.shape)

        neg_items_RL[neg_rand_indices] = neg_items_rand[neg_rand_indices.squeeze(0)]
        selected_neg_outfits_rules.unsqueeze(0)[neg_rand_indices] = neg_rules[neg_rand_indices.squeeze(0)]

        #neg_rules, selected_neg_outfits_rules)
        #user
        neg_rand_u_scores = torch.sum(u_e * neg_rand_u, dim=-1)
        neg_RL_u_scores = torch.sum(u_e * neg_RL_u, dim = -1)
        neg_rand_u_indices = torch.lt((neg_RL_u_scores - neg_rand_u_scores),0)
        neg_users_RL[neg_rand_u_indices] = neg_users_rand[neg_rand_u_indices.squeeze(0)]        
        return neg_items_RL, neg_users_RL, selected_neg_outfits_rules

    def _l2_loss(self, t):
        return torch.sum(t ** 2) / 2

    def inference(self, users):
        num_entity = self.n_nodes - self.n_users - self.n_items
        user_embed, item_embed, _ = torch.split(
            self.all_embed, [self.n_users, self.n_items, num_entity], dim=0
        )

        user_embed = user_embed[users]
        prediction = torch.matmul(user_embed, item_embed.t())
        return prediction
    def generate(self, edges_matrix):
        edges = self.build_edge(edges_matrix)
        x = self.all_embed
        gcn_embedding = self.gcn(x, edges.t().contiguous())
        return gcn_embedding
        
    def rank(self, users, items):
        u_e = self.all_embed[users]
        i_e = self.all_embed[items]

        u_e = u_e.unsqueeze(dim=1)
        ranking = torch.sum(u_e * i_e, dim=2)
        ranking = ranking.squeeze()

        return ranking

    def __str__(self):
        return "recommender using KGAT, embedding size {}".format(
            self.args_config.emb_size
        )


class KGAT_smodal(nn.Module):
    def __init__(self, data_config, args_config, triplets, modal):
        super(KGAT_smodal, self).__init__()
        self.args_config = args_config
        self.data_config = data_config
        self.n_users = data_config["n_users"]
        self.n_items = data_config["n_items"]
        self.n_nodes = data_config["n_nodes"]

        """set input and output channel manually"""
        input_channel = 64
        output_channel = 64
        self.gcn = GraphConv_KGAT(input_channel, output_channel)
        self.triplets = triplets
        self.img_dim_change = nn.Linear(512, 64)
        self.txt_dim_change = nn.Linear(384, 64)
        self.emb_size = args_config.dim
        self.regs = args_config.l2
        # self.regs = eval(args_config.regs)
        self.modal = modal
        self.all_embed = self._init_weight(triplets, modal)

    def _init_weight(self, triplets, modal):
        all_embed = nn.Parameter(
            torch.FloatTensor(self.n_nodes, self.emb_size), requires_grad=True
        )
        ui = self.n_users + self.n_items

        if self.args_config.pretrain_r:
            nn.init.xavier_uniform_(all_embed)
            all_embed.data[:ui] = self.data_config["all_embed"]
        else:
            nn.init.xavier_uniform_(all_embed)

        comprise_dict = {}
        for i in triplets:
            if i[1] == 1:
                outfit = i[0]
                item = i[2]
                if outfit not in comprise_dict:
                    comprise_dict[outfit] = [item]
                else:
                    comprise_dict[outfit].append(item)

        if modal == 'image':
            img_embedding = pickle.load(open(self.args_config.img_embedding_path, 'rb'))
            img_embedding[1] = self.img_dim_change(img_embedding[1])
            img_embedding[0] = [i + self.n_users for i in img_embedding[0]]
            if self.args_config.item:
                for num, i in enumerate(img_embedding[0]):
                    with torch.no_grad():
                        all_embed[i] = img_embedding[1][num]

            if self.args_config.outfit:
                for outfit in comprise_dict:
                    items = comprise_dict[outfit]
                    items_embedding = []
                    for i in items:
                        index = img_embedding[0].index(i)
                        items_embedding.append(img_embedding[1][index])
                    items_embedding = torch.stack(items_embedding)
                    with torch.no_grad():
                        all_embed[outfit] = torch.mean(items_embedding, dim=0)

            if self.args_config.outfit_item:
                for num, i in enumerate(img_embedding[0]):
                    with torch.no_grad():
                        all_embed[i] = img_embedding[1][num]

                for outfit in comprise_dict:
                    items = comprise_dict[outfit]
                    items_embedding = []
                    for i in items:
                        index = img_embedding[0].index(i)
                        items_embedding.append(img_embedding[1][index])
                    items_embedding = torch.stack(items_embedding)
                    with torch.no_grad():
                        all_embed[outfit] = torch.mean(items_embedding, dim=0)

        elif modal == 'txt':
            txt_embedding = pickle.load(open(self.args_config.txt_embedding_path, 'rb'))
            txt_embedding[1] = self.txt_dim_change(txt_embedding[1])
            txt_embedding[0] = [i + self.n_users for i in txt_embedding[0]]
            if self.args_config.item:
                for num, i in enumerate(txt_embedding[0]):
                    with torch.no_grad():
                        all_embed[i] = txt_embedding[1][num]

            if self.args_config.outfit:
                for outfit in comprise_dict:
                    items = comprise_dict[outfit]
                    items_embedding = []
                    for i in items:
                        index = txt_embedding[0].index(i)
                        items_embedding.append(txt_embedding[1][index])
                    items_embedding = torch.stack(items_embedding)
                    with torch.no_grad():
                        all_embed[outfit] = torch.mean(items_embedding, dim=0)

            if self.args_config.outfit_item:
                for num, i in enumerate(txt_embedding[0]):
                    with torch.no_grad():
                        all_embed[i] = txt_embedding[1][num]

                for outfit in comprise_dict:
                    items = comprise_dict[outfit]
                    items_embedding = []
                    for i in items:
                        index = txt_embedding[0].index(i)
                        items_embedding.append(txt_embedding[1][index])
                    items_embedding = torch.stack(items_embedding)
                    with torch.no_grad():
                        all_embed[outfit] = torch.mean(items_embedding, dim=0)

        return all_embed

    def build_edge(self, adj_matrix):
        """build edges based on adj_matrix"""
        sample_edge = self.args_config.edge_threshold
        edge_matrix = adj_matrix

        n_node = edge_matrix.size(0)
        node_index = (
            torch.arange(n_node, device=edge_matrix.device)
                .unsqueeze(1)
                .repeat(1, sample_edge)
                .flatten()
        )
        neighbor_index = edge_matrix.flatten()
        edges = torch.cat((node_index.unsqueeze(1), neighbor_index.unsqueeze(1)), dim=1)
        return edges

    def forward(self, user, pos_item, neg_item, edges_matrix, pos_rules, selected_neg_outfits_rules, flag_debug, epoch,
                flag_user):
        u_e, pos_e, neg_e, pos_rule_e, neg_rule_e = (
            self.all_embed[user],
            self.all_embed[pos_item],
            self.all_embed[neg_item],
            self.all_embed[pos_rules],
            self.all_embed[selected_neg_outfits_rules],
        )

        edges = self.build_edge(edges_matrix)
        x = self.all_embed
        # 182423
        gcn_embedding = self.gcn(x, edges.t().contiguous())

        u_e_, pos_e_, neg_e_, pos_rule_e_, neg_rule_e_ = (
            gcn_embedding[user],
            gcn_embedding[pos_item],
            gcn_embedding[neg_item],
            gcn_embedding[pos_rules],
            gcn_embedding[selected_neg_outfits_rules],
        )

        # u_e = torch.cat([u_e, u_e_], dim=1)
        # pos_e = torch.cat([pos_e, pos_e_], dim=1)
        # pos_rule_e = torch.cat([pos_rule_e, pos_rule_e_], dim=1)
        # # neg_e = torch.cat([neg_e, neg_e_], dim=1)
        # pos_scores = torch.sum(u_e * pos_e, dim=1)
        # pos_rule_scores = torch.sum(u_e * pos_rule_e, dim=1)
        # if self.args_config.score:
        #     if self.args_config.score_change:
        #         if epoch < 60:
        #             pos_scores += self.args_config.rulescore_pos * pos_rule_scores
        #         else:
        #             pos_scores += self.args_config.rulescore_pos * pos_rule_scores * 10
        #     else:
        #         pos_scores += self.args_config.rulescore_pos * pos_rule_scores
        #
        # '''
        # if flag_debug == True:
        #     pdb.set_trace()
        # '''
        # if flag_user == False:
        #     neg_e = torch.cat([neg_e, neg_e_], dim=1)
        #     neg_rule_e = torch.cat([neg_rule_e, neg_rule_e_], dim=1)
        #     reg_loss = self._l2_loss(u_e) + self._l2_loss(pos_e) + self._l2_loss(neg_e) + 0.001 * self._l2_loss(
        #         pos_rule_e) + 0.001 * self._l2_loss(neg_rule_e)
        #     neg_scores = torch.sum(u_e * neg_e, dim=1)
        #     neg_rule_scores = torch.sum(u_e * neg_rule_e, dim=1)
        #     if self.args_config.score:
        #         if self.args_config.score_change:
        #             if epoch < 60:
        #                 neg_scores += self.args_config.rulescore_neg * neg_rule_scores
        #             else:
        #                 neg_scores += self.args_config.rulescore_neg * neg_rule_scores * 0.1
        #         else:
        #             neg_scores += self.args_config.rulescore_neg * neg_rule_scores
        #
        #
        # else:
        #     neg_user_e = torch.cat([neg_e, neg_e_], dim=1)
        #     reg_loss = self._l2_loss(u_e) + self._l2_loss(pos_e) + self._l2_loss(neg_user_e)
        #     neg_scores = torch.sum(neg_user_e * pos_e, dim=1)
        #     neg_rule_scores = torch.sum(neg_user_e * pos_rule_e, dim=1)
        #     if self.args_config.score:
        #         if self.args_config.score_change:
        #             if epoch < 60:
        #                 neg_scores += self.args_config.rulescore_neg * neg_rule_scores
        #             else:
        #                 neg_scores += self.args_config.rulescore_neg * neg_rule_scores * 0.1
        #         else:
        #             neg_scores += self.args_config.rulescore_neg * neg_rule_scores
        #
        #     # neg_rule_scores = torch.sum(neg_user_e * pos_rule_e, dim=1)
        #     # neg_scores += neg_rule_scores
        #
        # # neg_scores = torch.sum(u_e * neg_e, dim=1)
        #
        # # Defining objective function that contains:
        # # ... (1) bpr loss
        # bpr_loss = torch.log(torch.sigmoid(pos_scores - neg_scores))
        # bpr_loss = -torch.mean(bpr_loss)
        #
        # # ... (2) emb loss
        # reg_loss = self._l2_loss(u_e) + self._l2_loss(pos_e) + self._l2_loss(neg_e)
        # reg_loss = self.regs * reg_loss
        return u_e, pos_e, neg_e, pos_rule_e, neg_rule_e, u_e_, pos_e_, neg_e_, pos_rule_e_, neg_rule_e_, x


    def get_reward(self, users, pos_items, neg_items):
        u_e = self.all_embed[users]
        pos_e = self.all_embed[pos_items]
        neg_e = self.all_embed[neg_items]

        neg_scores = torch.sum(u_e * neg_e, dim=-1)
        ij = torch.sum(neg_e * pos_e, dim=-1)
        neg_scores_2 = - torch.sum(torch.sigmoid(-1e3 * u_e * neg_e), dim=-1)
        # reward = 1e-6*neg_scores_2
        # reward = neg_scores + ij + 1e-3*neg_scores_2
        reward = 1e+4 * (neg_scores + ij) + 1e-1 * neg_scores_2
        return reward

    def get_samples(self, users, neg_users_rand, neg_users_RL, pos_items, neg_items_rand, neg_items_RL, pos_rules,
                    neg_rules, selected_neg_outfits_rules):
        u_e = self.all_embed[users]
        pos_e = self.all_embed[pos_items]
        neg_rand_e = self.all_embed[neg_items_rand]
        neg_RL_e = self.all_embed[neg_items_RL]
        neg_rules_e = self.all_embed[neg_rules]
        neg_RL_rule_e = self.all_embed[selected_neg_outfits_rules]
        neg_rand_u = self.all_embed[neg_users_rand]
        neg_RL_u = self.all_embed[neg_users_RL]
        # outfit
        a = u_e * neg_rand_e
        neg_rand_scores = torch.sum(u_e * neg_rand_e, dim=-1)
        neg_RL_scores = torch.sum(u_e * neg_RL_e, dim=-1)
        # rule score
        neg_rand_rule_scores = torch.sum(u_e * neg_rules_e, dim=1)
        neg_RL_rule_scores = torch.sum(u_e * neg_RL_rule_e, dim=-1)
        if self.args_config.score:
            neg_RL_scores += 0.001 * neg_RL_rule_scores
            neg_rand_scores += 0.001 * neg_rand_rule_scores
        neg_rand_indices = torch.lt((neg_RL_scores - neg_rand_scores), 0)

        # print(neg_rand_indices.shape)

        neg_items_RL[neg_rand_indices] = neg_items_rand[neg_rand_indices.squeeze(0)]
        selected_neg_outfits_rules.unsqueeze(0)[neg_rand_indices] = neg_rules[neg_rand_indices.squeeze(0)]

        # neg_rules, selected_neg_outfits_rules)
        # user
        neg_rand_u_scores = torch.sum(u_e * neg_rand_u, dim=-1)
        neg_RL_u_scores = torch.sum(u_e * neg_RL_u, dim=-1)
        neg_rand_u_indices = torch.lt((neg_RL_u_scores - neg_rand_u_scores), 0)
        neg_users_RL[neg_rand_u_indices] = neg_users_rand[neg_rand_u_indices.squeeze(0)]
        return neg_items_RL, neg_users_RL, selected_neg_outfits_rules

    def _l2_loss(self, t):
        return torch.sum(t ** 2) / 2

    def inference(self, users):
        num_entity = self.n_nodes - self.n_users - self.n_items
        user_embed, item_embed, _ = torch.split(
            self.all_embed, [self.n_users, self.n_items, num_entity], dim=0
        )

        user_embed = user_embed[users]
        prediction = torch.matmul(user_embed, item_embed.t())
        return prediction

    def generate(self, edges_matrix):
        edges = self.build_edge(edges_matrix)
        x = self.all_embed
        gcn_embedding = self.gcn(x, edges.t().contiguous())
        return gcn_embedding

    def rank(self, users, items):
        u_e = self.all_embed[users]
        i_e = self.all_embed[items]

        u_e = u_e.unsqueeze(dim=1)
        ranking = torch.sum(u_e * i_e, dim=2)
        ranking = ranking.squeeze()

        return ranking

    def __str__(self):
        return "recommender using KGAT, embedding size {}".format(
            self.args_config.emb_size
        )


class KGAT_MM(nn.Module):
    def __init__(self, data_config, args_config, triplets):
        super(KGAT_MM, self).__init__()
        self.args_config = args_config
        self.data_config = data_config
        self.n_users = data_config["n_users"]
        self.n_items = data_config["n_items"]
        self.n_nodes = data_config["n_nodes"]

        """set input and output channel manually"""
        self.triplets = triplets
        self.emb_size = args_config.dim
        self.regs = args_config.l2
        # self.regs = eval(args_config.regs)


    def build_edge(self, adj_matrix):
        """build edges based on adj_matrix"""
        sample_edge = self.args_config.edge_threshold
        edge_matrix = adj_matrix

        n_node = edge_matrix.size(0)
        node_index = (
            torch.arange(n_node, device=edge_matrix.device)
                .unsqueeze(1)
                .repeat(1, sample_edge)
                .flatten()
        )
        neighbor_index = edge_matrix.flatten()
        edges = torch.cat((node_index.unsqueeze(1), neighbor_index.unsqueeze(1)), dim=1)
        return edges

    def forward(self, entity_u_e, entity_pos_e, entity_neg_e, entity_pos_rule_e, entity_neg_rule_e, entity_u_e_, entity_pos_e_, entity_neg_e_, entity_pos_rule_e_, entity_neg_rule_e_ ,
                 img_u_e, img_pos_e, img_neg_e, img_pos_rule_e, img_neg_rule_e, img_u_e_, img_pos_e_, img_neg_e_, img_pos_rule_e_, img_neg_rule_e_ ,
                 txt_u_e, txt_pos_e, txt_neg_e, txt_pos_rule_e, txt_neg_rule_e, txt_u_e_, txt_pos_e_, txt_neg_e_, txt_pos_rule_e_, txt_neg_rule_e_ ,
                entity_all_embedding,img_all_embedding, txt_all_embedding,edges_matrix, flag_debug, epoch,flag_user):

        self.all_embed = (entity_all_embedding + img_all_embedding + txt_all_embedding)/3

        u_e = (entity_u_e + img_u_e + txt_u_e)/3
        u_e_ = (entity_u_e_ + img_u_e_ + txt_u_e_)/3
        pos_e = (entity_pos_e + img_pos_e + txt_pos_e)/3
        pos_e_ = (entity_pos_e_ + img_pos_e_ + txt_pos_e_)/3
        pos_rule_e = (entity_pos_rule_e + img_pos_rule_e + txt_pos_rule_e)/3
        pos_rule_e_ = (entity_pos_rule_e_ + img_pos_rule_e_ + txt_pos_rule_e_)/3

        neg_e = (entity_neg_e + img_neg_e + txt_neg_e)/3
        neg_e_ = (entity_neg_e_ + img_neg_e_ + txt_neg_e_)/3
        neg_rule_e = (entity_neg_rule_e + img_neg_rule_e + txt_neg_rule_e)/3
        neg_rule_e_ = (entity_neg_rule_e_ + img_neg_rule_e_ + txt_neg_rule_e_)/3

        u_e = torch.cat([u_e, u_e_], dim=1)
        pos_e = torch.cat([pos_e, pos_e_], dim=1)
        pos_rule_e = torch.cat([pos_rule_e, pos_rule_e_], dim=1)
        # neg_e = torch.cat([neg_e, neg_e_], dim=1)
        pos_scores = torch.sum(u_e * pos_e, dim=1)
        pos_rule_scores = torch.sum(u_e * pos_rule_e, dim=1)
        if self.args_config.score:
            if self.args_config.score_change:
                if epoch < 60:
                    pos_scores += self.args_config.rulescore_pos * pos_rule_scores
                else:
                    pos_scores += self.args_config.rulescore_pos * pos_rule_scores * 10
            else:
                pos_scores += self.args_config.rulescore_pos * pos_rule_scores

        '''
        if flag_debug == True:
            pdb.set_trace()
        '''
        if flag_user == False:
            neg_e = torch.cat([neg_e, neg_e_], dim=1)
            neg_rule_e = torch.cat([neg_rule_e, neg_rule_e_], dim=1)
            reg_loss = self._l2_loss(u_e) + self._l2_loss(pos_e) + self._l2_loss(neg_e) + 0.001 * self._l2_loss(
                pos_rule_e) + 0.001 * self._l2_loss(neg_rule_e)
            neg_scores = torch.sum(u_e * neg_e, dim=1)
            neg_rule_scores = torch.sum(u_e * neg_rule_e, dim=1)
            if self.args_config.score:
                if self.args_config.score_change:
                    if epoch < 60:
                        neg_scores += self.args_config.rulescore_neg * neg_rule_scores
                    else:
                        neg_scores += self.args_config.rulescore_neg * neg_rule_scores * 0.1
                else:
                    neg_scores += self.args_config.rulescore_neg * neg_rule_scores


        else:
            neg_user_e = torch.cat([neg_e, neg_e_], dim=1)
            reg_loss = self._l2_loss(u_e) + self._l2_loss(pos_e) + self._l2_loss(neg_user_e)
            neg_scores = torch.sum(neg_user_e * pos_e, dim=1)
            neg_rule_scores = torch.sum(neg_user_e * pos_rule_e, dim=1)
            if self.args_config.score:
                if self.args_config.score_change:
                    if epoch < 60:
                        neg_scores += self.args_config.rulescore_neg * neg_rule_scores
                    else:
                        neg_scores += self.args_config.rulescore_neg * neg_rule_scores * 0.1
                else:
                    neg_scores += self.args_config.rulescore_neg * neg_rule_scores

            # neg_rule_scores = torch.sum(neg_user_e * pos_rule_e, dim=1)
            # neg_scores += neg_rule_scores

        # neg_scores = torch.sum(u_e * neg_e, dim=1)

        # Defining objective function that contains:
        # ... (1) bpr loss
        bpr_loss = torch.log(torch.sigmoid(pos_scores - neg_scores))
        bpr_loss = -torch.mean(bpr_loss)

        # ... (2) emb loss
        reg_loss = self._l2_loss(u_e) + self._l2_loss(pos_e) + self._l2_loss(neg_e)
        reg_loss = self.regs * reg_loss
        return bpr_loss, reg_loss
        # loss = bpr_loss + reg_loss
        # return loss, bpr_loss, reg_loss

    def get_reward(self, users, pos_items, neg_items):
        u_e = self.all_embed[users]
        pos_e = self.all_embed[pos_items]
        neg_e = self.all_embed[neg_items]

        neg_scores = torch.sum(u_e * neg_e, dim=-1)
        ij = torch.sum(neg_e * pos_e, dim=-1)
        neg_scores_2 = - torch.sum(torch.sigmoid(-1e3 * u_e * neg_e), dim=-1)
        # reward = 1e-6*neg_scores_2
        # reward = neg_scores + ij + 1e-3*neg_scores_2
        reward = 1e+4 * (neg_scores + ij) + 1e-1 * neg_scores_2
        return reward

    def get_samples(self, users, neg_users_rand, neg_users_RL, pos_items, neg_items_rand, neg_items_RL, pos_rules,
                    neg_rules, selected_neg_outfits_rules):
        u_e = self.all_embed[users]
        pos_e = self.all_embed[pos_items]
        neg_rand_e = self.all_embed[neg_items_rand]
        neg_RL_e = self.all_embed[neg_items_RL]
        neg_rules_e = self.all_embed[neg_rules]
        neg_RL_rule_e = self.all_embed[selected_neg_outfits_rules]
        neg_rand_u = self.all_embed[neg_users_rand]
        neg_RL_u = self.all_embed[neg_users_RL]
        # outfit
        a = u_e * neg_rand_e
        neg_rand_scores = torch.sum(u_e * neg_rand_e, dim=-1)
        neg_RL_scores = torch.sum(u_e * neg_RL_e, dim=-1)
        # rule score
        neg_rand_rule_scores = torch.sum(u_e * neg_rules_e, dim=1)
        neg_RL_rule_scores = torch.sum(u_e * neg_RL_rule_e, dim=-1)
        if self.args_config.score:
            neg_RL_scores += 0.001 * neg_RL_rule_scores
            neg_rand_scores += 0.001 * neg_rand_rule_scores
        neg_rand_indices = torch.lt((neg_RL_scores - neg_rand_scores), 0)

        # print(neg_rand_indices.shape)

        neg_items_RL[neg_rand_indices] = neg_items_rand[neg_rand_indices.squeeze(0)]
        selected_neg_outfits_rules.unsqueeze(0)[neg_rand_indices] = neg_rules[neg_rand_indices.squeeze(0)]

        # neg_rules, selected_neg_outfits_rules)
        # user
        neg_rand_u_scores = torch.sum(u_e * neg_rand_u, dim=-1)
        neg_RL_u_scores = torch.sum(u_e * neg_RL_u, dim=-1)
        neg_rand_u_indices = torch.lt((neg_RL_u_scores - neg_rand_u_scores), 0)
        neg_users_RL[neg_rand_u_indices] = neg_users_rand[neg_rand_u_indices.squeeze(0)]
        return neg_items_RL, neg_users_RL, selected_neg_outfits_rules

    def _l2_loss(self, t):
        return torch.sum(t ** 2) / 2

    def inference(self, users):
        num_entity = self.n_nodes - self.n_users - self.n_items
        user_embed, item_embed, _ = torch.split(
            self.all_embed, [self.n_users, self.n_items, num_entity], dim=0
        )

        user_embed = user_embed[users]
        prediction = torch.matmul(user_embed, item_embed.t())
        return prediction

    def generate(self, edges_matrix):
        edges = self.build_edge(edges_matrix)
        x = self.all_embed
        gcn_embedding = self.gcn(x, edges.t().contiguous())
        return gcn_embedding

    def rank(self, users, items):
        u_e = self.all_embed[users]
        i_e = self.all_embed[items]

        u_e = u_e.unsqueeze(dim=1)
        ranking = torch.sum(u_e * i_e, dim=2)
        ranking = ranking.squeeze()

        return ranking

    def __str__(self):
        return "recommender using KGAT, embedding size {}".format(
            self.args_config.emb_size
        )


class KGPolicy_smodal(nn.Module):
    def __init__(self, dis, params, config, triplets, modal):
        super(KGPolicy_smodal, self).__init__()
        self.config = config
        self.data_config = params
        self.n_users = params["n_users"]
        self.n_items = params["n_items"]
        self.n_nodes = params["n_nodes"]
        self.item_range = params["item_range"]
        self.dis = dis

        """set input and output channel manually"""
        input_channel = 64
        output_channel = 64
        self.gcn = GraphConv_KGAT(input_channel, output_channel)
        self.triplets = triplets
        self.img_dim_change = nn.Linear(512, 64)
        self.txt_dim_change = nn.Linear(384, 64)
        self.emb_size = config.dim
        self.regs = config.l2
        # self.regs = eval(args_config.regs)

        self.all_embed = self._init_weight(triplets, modal)

    def _init_weight(self, triplets, modal):
        all_embed = nn.Parameter(
            torch.FloatTensor(self.n_nodes, self.emb_size), requires_grad=True
        )
        ui = self.n_users + self.n_items

        if self.config.pretrain_r:
            nn.init.xavier_uniform_(all_embed)
            all_embed.data[:ui] = self.data_config["all_embed"]
        else:
            nn.init.xavier_uniform_(all_embed)

        comprise_dict = {}
        for i in triplets:
            if i[1] == 1:
                outfit = i[0]
                item = i[2]
                if outfit not in comprise_dict:
                    comprise_dict[outfit] = [item]
                else:
                    comprise_dict[outfit].append(item)

        if modal == 'image':
            img_embedding = pickle.load(open(self.config.img_embedding_path, 'rb'))
            img_embedding[1] = self.img_dim_change(img_embedding[1])
            img_embedding[0] = [i + self.n_users for i in img_embedding[0]]
            if self.config.item:
                for num, i in enumerate(img_embedding[0]):
                    with torch.no_grad():
                        all_embed[i] = img_embedding[1][num]

            if self.config.outfit:
                for outfit in comprise_dict:
                    items = comprise_dict[outfit]
                    items_embedding = []
                    for i in items:
                        index = img_embedding[0].index(i)
                        items_embedding.append(img_embedding[1][index])
                    items_embedding = torch.stack(items_embedding)
                    with torch.no_grad():
                        all_embed[outfit] = torch.mean(items_embedding, dim=0)

            if self.config.outfit_item:
                for num, i in enumerate(img_embedding[0]):
                    with torch.no_grad():
                        all_embed[i] = img_embedding[1][num]

                for outfit in comprise_dict:
                    items = comprise_dict[outfit]
                    items_embedding = []
                    for i in items:
                        index = img_embedding[0].index(i)
                        items_embedding.append(img_embedding[1][index])
                    items_embedding = torch.stack(items_embedding)
                    with torch.no_grad():
                        all_embed[outfit] = torch.mean(items_embedding, dim=0)

        elif modal == 'txt':
            txt_embedding = pickle.load(open(self.config.txt_embedding_path, 'rb'))
            txt_embedding[1] = self.txt_dim_change(txt_embedding[1])
            txt_embedding[0] = [i + self.n_users for i in txt_embedding[0]]
            if self.config.item:
                for num, i in enumerate(txt_embedding[0]):
                    with torch.no_grad():
                        all_embed[i] = txt_embedding[1][num]

            if self.config.outfit:
                for outfit in comprise_dict:
                    items = comprise_dict[outfit]
                    items_embedding = []
                    for i in items:
                        index = txt_embedding[0].index(i)
                        items_embedding.append(txt_embedding[1][index])
                    items_embedding = torch.stack(items_embedding)
                    with torch.no_grad():
                        all_embed[outfit] = torch.mean(items_embedding, dim=0)

            if self.config.outfit_item:
                for num, i in enumerate(txt_embedding[0]):
                    with torch.no_grad():
                        all_embed[i] = txt_embedding[1][num]

                for outfit in comprise_dict:
                    items = comprise_dict[outfit]
                    items_embedding = []
                    for i in items:
                        index = txt_embedding[0].index(i)
                        items_embedding.append(txt_embedding[1][index])
                    items_embedding = torch.stack(items_embedding)
                    with torch.no_grad():
                        all_embed[outfit] = torch.mean(items_embedding, dim=0)

        return all_embed

    def build_edge(self, adj_matrix):
        """build edges based on adj_matrix"""
        sample_edge = self.config.edge_threshold
        edge_matrix = adj_matrix

        n_node = edge_matrix.size(0)
        node_index = (
            torch.arange(n_node, device=edge_matrix.device)
                .unsqueeze(1)
                .repeat(1, sample_edge)
                .flatten()
        )
        neighbor_index = edge_matrix.flatten()
        edges = torch.cat((node_index.unsqueeze(1), neighbor_index.unsqueeze(1)), dim=1)
        return edges

    def forward(self, data_batch, adj_matrix, edge_matrix):
        # def forward(self, data_batch, adj_matrix, edge_matrix):
        users = data_batch["users"]
        pos = data_batch["pos_items"]
        self.edges = self.build_edge(edge_matrix)

        neg_list = torch.tensor([], dtype=torch.long, device=adj_matrix.device)
        prob_list = torch.tensor([], device=adj_matrix.device)
        neg_list_user = torch.tensor([], dtype=torch.long, device=adj_matrix.device)
        prob_list_user = torch.tensor([], device=adj_matrix.device)

        k = self.config.k_step
        assert k > 0
        for _ in range(k):
            # OUTFIT
            candidate_neg, two_hop_logits = self.kg_step_outfit(pos, users, adj_matrix, step=2)
            candidate_neg = self.filter_entity(candidate_neg, self.item_range)
            good_neg, good_logits = self.prune_step(
                self.dis, candidate_neg, users, two_hop_logits
            )
            good_logits = good_logits
            good_neg = good_neg

            neg_list = torch.cat([neg_list, good_neg.unsqueeze(0)])
            prob_list = torch.cat([prob_list, good_logits.unsqueeze(0)])

            # USER
            candidate_neg_user, one_hop_logits = self.kg_step_user(pos, users, adj_matrix, step=1)
            candidate_neg_user = self.filter_entity_user(candidate_neg_user, self.n_users)

            good_neg_user = candidate_neg_user
            good_logits_user = one_hop_logits

            neg_list_user = torch.cat([neg_list_user, good_neg_user.unsqueeze(0)])
            prob_list_user = torch.cat([prob_list_user, good_logits_user.unsqueeze(0)])

        return neg_list, prob_list, neg_list_user, prob_list_user

        '''
        for _ in range(k):
            """sample candidate negative items based on knowledge graph"""
            #one_hop, one_hop_logits = self.kg_step_outfit(pos, users, adj_matrix, step=1)

            one_hop_item, one_hop_logits = self.kg_step_outfit2item(pos, users, adj_matrix, step=1)

            """
            flag_one_hop_list = []
            count_0_step1 = 0 
            for temp_node in range(len(one_hop)):
                flag_one_hop = (one_hop[temp_node]>=114728 and one_hop[temp_node]<=139689).item()
                flag_one_hop_list.append(flag_one_hop)
                if flag_one_hop==0:
                    count_0_step1 += 1
            """

            candidate_neg, two_hop_logits = self.kg_step_outfit(
                one_hop_item, users, adj_matrix, step=2
            )

            """
            flag_two_hop_list = []
            ratio_list =  []
            for temp_node_indx in range(candidate_neg.shape[0]):
                count_0_step2 = 0
                temp_node_list = candidate_neg[temp_node_indx]
                for temp_node in range(len(temp_node_list)):
                    flag_two_hop = (temp_node_list[temp_node]>=114728 and temp_node_list[temp_node]<=139689).item()
                    flag_two_hop_list.append(flag_two_hop)
                    if flag_two_hop==0:
                        count_0_step2 += 1
                ratio_list.append((count_0_step2/len(temp_node_list)))
            pdb.set_trace()
            """

            candidate_neg = self.filter_entity(candidate_neg, self.item_range)
            good_neg, good_logits = self.prune_step(
                self.dis, candidate_neg, users, two_hop_logits
            )
            good_logits = good_logits + one_hop_logits

            neg_list = torch.cat([neg_list, good_neg.unsqueeze(0)])
            prob_list = torch.cat([prob_list, good_logits.unsqueeze(0)])

            pos = good_neg

        return neg_list, prob_list, neg_list, prob_list
        '''

        """
        one_hop, one_hop_logits = self.kg_step(pos, users, adj_matrix, step=1)

        candidate_neg, two_hop_logits = self.kg_step(
            one_hop, users, adj_matrix, step=2
        )
        candidate_neg = self.filter_entity(candidate_neg, self.item_range)
        good_neg, good_logits = self.prune_step(
            self.dis, candidate_neg, users, two_hop_logits
        )
        good_logits = good_logits + one_hop_logits

        neg_list = torch.cat([neg_list, good_neg.unsqueeze(0)])
        prob_list = torch.cat([prob_list, good_logits.unsqueeze(0)])
        pos = good_neg
        """

        '''
        candidate_neg, two_hop_logits = self.kg_step_outfit(pos, users, adj_matrix, step=2)
        candidate_neg = self.filter_entity(candidate_neg, self.item_range)
        good_neg, good_logits = self.prune_step(
            self.dis, candidate_neg, users, two_hop_logits
        )
        good_logits = good_logits
        good_neg = good_neg

        neg_list = torch.cat([neg_list, good_neg.unsqueeze(0)])
        prob_list = torch.cat([prob_list, good_logits.unsqueeze(0)])
        '''

        '''
        k = self.config.k_step
        assert k > 0
        for _ in range(k):
            #OUTFIT 
            candidate_neg, two_hop_logits = self.kg_step_outfit(pos, users, adj_matrix, step=2)
            candidate_neg = self.filter_entity(candidate_neg, self.item_range)
            good_neg, good_logits = self.prune_step(
                self.dis, candidate_neg, users, two_hop_logits
            )
            good_logits = good_logits
            good_neg = good_neg

            neg_list = torch.cat([neg_list, good_neg.unsqueeze(0)])
            prob_list = torch.cat([prob_list, good_logits.unsqueeze(0)])

            #USER
            candidate_neg_user, one_hop_logits = self.kg_step_user(pos, users, adj_matrix, step=1)
            candidate_neg_user = self.filter_entity_user(candidate_neg_user, self.n_users)

            good_neg_user = candidate_neg_user
            good_logits_user = one_hop_logits

            neg_list_user = torch.cat([neg_list_user, good_neg_user.unsqueeze(0)])
            prob_list_user = torch.cat([prob_list_user, good_logits_user.unsqueeze(0)])

        return neg_list, prob_list, neg_list_user, prob_list_user
        '''

    def kg_step_user(self, pos, user, adj_matrix, step):
        # x = self.entity_embedding
        edges = self.edges
        x = self.all_embed
        gcn_embedding = self.gcn(x, edges.t().contiguous())
        # gcn_embedding = torch.cat([user_gcn_emb,entity_gcn_emb], dim=0)

        """knowledge graph embedding using gcn"""
        # gcn_embedding = torch.cat([entity_gcn_emb,user_gcn_emb], dim=0)
        # gcn_embedding = self.gcn(x, edges.t().contiguous())

        """use knowledge embedding to decide candidate negative items"""
        u_e = gcn_embedding[user]
        u_e = u_e.unsqueeze(dim=1)
        pos_e = gcn_embedding[pos]
        pos_e = pos_e.unsqueeze(dim=2)

        one_hop_user = adj_matrix[user]
        i_user_e = gcn_embedding[one_hop_user]

        p_entity = F.leaky_relu(u_e * i_user_e)
        p = torch.sum(p_entity, dim=-1)
        '''
        p = torch.matmul(p_entity, pos_e)
        p = p.squeeze()
        '''
        logits = F.softmax(p, dim=1)

        """sample negative items based on logits"""
        batch_size = logits.size(0)
        if step == 1:
            nid = torch.argmax(logits, dim=1, keepdim=True)
        else:
            n = self.config.num_sample
            _, indices = torch.sort(logits, descending=True)
            nid = indices[:, :n]
        row_id = torch.arange(batch_size, device=logits.device).unsqueeze(1)

        candidate_neg = one_hop_user[row_id, nid].squeeze()
        candidate_logits = torch.log(logits[row_id, nid]).squeeze()

        return candidate_neg, candidate_logits

    def kg_step_outfit2item(self, pos, user, adj_matrix, step):
        # x = self.entity_embedding
        edges = self.edges
        x = self.all_embed
        gcn_embedding = self.gcn(x, edges.t().contiguous())
        # gcn_embedding = self.gcn(x, edges.t().contiguous())

        """use knowledge embedding to decide candidate negative items"""
        u_e = gcn_embedding[user]
        u_e = u_e.unsqueeze(dim=2)
        pos_e = gcn_embedding[pos]
        pos_e = pos_e.unsqueeze(dim=1)

        one_hop = adj_matrix[pos]
        # one_hop = self.filter_entity_item(one_hop, self.item_range)
        # 139689-185310
        i_e = gcn_embedding[one_hop]

        p_entity = F.leaky_relu(pos_e * i_e)
        p = torch.sum(p_entity, dim=-1)
        '''
        p = torch.matmul(p_entity, u_e)
        p = p.squeeze()
        '''
        logits = F.softmax(p, dim=1)

        """sample negative items based on logits"""
        batch_size = logits.size(0)
        if step == 1:
            nid = torch.argmax(logits, dim=1, keepdim=True)
        else:
            n = self.config.num_sample
            _, indices = torch.sort(logits, descending=True)
            nid = indices[:, :n]
        row_id = torch.arange(batch_size, device=logits.device).unsqueeze(1)

        candidate_neg = one_hop[row_id, nid].squeeze()
        candidate_logits = torch.log(logits[row_id, nid]).squeeze()

        return candidate_neg, candidate_logits

    def kg_step_outfit(self, pos, user, adj_matrix, step):
        # x = self.entity_embedding
        edges = self.edges
        x = self.all_embed
        gcn_embedding = self.gcn(x, edges.t().contiguous())
        # gcn_embedding = self.gcn(x, edges.t().contiguous())

        """use knowledge embedding to decide candidate negative items"""
        u_e = gcn_embedding[user]
        u_e = u_e.unsqueeze(dim=2)
        pos_e = gcn_embedding[pos]
        pos_e = pos_e.unsqueeze(dim=1)

        one_hop = adj_matrix[pos]
        i_e = gcn_embedding[one_hop]

        p_entity = F.leaky_relu(pos_e * i_e)
        p = torch.sum(p_entity, dim=-1)
        '''
        p = torch.matmul(p_entity, u_e)
        p = p.squeeze()
        '''
        logits = F.softmax(p, dim=1)

        """sample negative items based on logits"""
        batch_size = logits.size(0)
        if step == 1:
            nid = torch.argmax(logits, dim=1, keepdim=True)
        else:
            n = self.config.num_sample
            _, indices = torch.sort(logits, descending=True)
            nid = indices[:, :n]
        row_id = torch.arange(batch_size, device=logits.device).unsqueeze(1)

        candidate_neg = one_hop[row_id, nid].squeeze()
        candidate_logits = torch.log(logits[row_id, nid]).squeeze()

        return candidate_neg, candidate_logits

    def kg_step(self, pos, user, adj_matrix, step):
        # x = self.entity_embedding
        edges = self.edges
        x = self.all_embed
        gcn_embedding = self.gcn(x, edges.t().contiguous())
        """knowledge graph embedding using gcn"""
        # gcn_embedding = torch.cat([entity_gcn_emb,user_gcn_emb], dim=0)
        # gcn_embedding = self.gcn(x, edges.t().contiguous())

        """use knowledge embedding to decide candidate negative items"""
        u_e = gcn_embedding[user]
        u_e = u_e.unsqueeze(dim=2)
        pos_e = gcn_embedding[pos]
        pos_e = pos_e.unsqueeze(dim=1)

        one_hop = adj_matrix[pos]
        i_e = gcn_embedding[one_hop]

        p_entity = F.leaky_relu(pos_e * i_e)
        p = torch.matmul(p_entity, u_e)
        p = p.squeeze()
        logits = F.softmax(p, dim=1)

        """sample negative items based on logits"""
        batch_size = logits.size(0)
        if step == 1:
            nid = torch.argmax(logits, dim=1, keepdim=True)
        else:
            n = self.config.num_sample
            _, indices = torch.sort(logits, descending=True)
            nid = indices[:, :n]
        row_id = torch.arange(batch_size, device=logits.device).unsqueeze(1)

        candidate_neg = one_hop[row_id, nid].squeeze()
        candidate_logits = torch.log(logits[row_id, nid]).squeeze()

        return candidate_neg, candidate_logits

    def filter_entity_item(self, neg, item_range):
        low_range = int(item_range[1] + 1)
        high_range = self.n_nodes
        random_neg = torch.randint(low_range, high_range, neg.size(), device=neg.device)

        neg[neg > high_range] = random_neg[neg > high_range]
        neg[neg < low_range] = random_neg[neg < low_range]

        '''
        random_neg = torch.randint(
            int(item_range[0]), int(item_range[1] + 1), neg.size(), device=neg.device
        )
        neg[neg > item_range[1]] = random_neg[neg > item_range[1]]
        neg[neg < item_range[0]] = random_neg[neg < item_range[0]]
        '''
        return neg

    @staticmethod
    def prune_step(dis, negs, users, logits):
        with torch.no_grad():
            # pdb.set_trace()
            ranking = dis.rank(users, negs)

        """get most qualified negative item based on user-neg similarity"""
        indices = torch.argmax(ranking, dim=1)

        batch_size = negs.size(0)
        row_id = torch.arange(batch_size, device=negs.device).unsqueeze(1)
        indices = indices.unsqueeze(1)

        good_neg = negs[row_id, indices].squeeze()
        goog_logits = logits[row_id, indices].squeeze()

        return good_neg, goog_logits

    @staticmethod
    def filter_entity(neg, item_range):
        random_neg = torch.randint(
            int(item_range[0]), int(item_range[1] + 1), neg.size(), device=neg.device
        )
        neg[neg > item_range[1]] = random_neg[neg > item_range[1]]
        neg[neg < item_range[0]] = random_neg[neg < item_range[0]]
        return neg

    @staticmethod
    def filter_entity_outfit(neg, item_range):
        random_neg = torch.randint(
            int(item_range[0]), int(item_range[1] + 1), neg.size(), device=neg.device
        )
        neg[neg > item_range[1]] = random_neg[neg > item_range[1]]
        neg[neg < item_range[0]] = random_neg[neg < item_range[0]]
        return neg

    @staticmethod
    def filter_entity_user(neg, n_users):
        random_neg = torch.randint(0, n_users, neg.size(), device=neg.device
                                   )
        neg[neg > n_users] = random_neg[neg > n_users]
        neg[neg < 0] = random_neg[neg < 0]
        return neg


class KGPolicy(nn.Module):
    def __init__(self, dis, params, config,triplets):
        super(KGPolicy, self).__init__()
        self.config = config
        self.data_config = params 
        self.n_users = params["n_users"]
        self.n_items = params["n_items"]
        self.n_nodes = params["n_nodes"]
        self.item_range = params["item_range"]
        self.dis = dis

        """set input and output channel manually"""
        input_channel = 64
        output_channel = 64
        self.gcn = GraphConv_KGAT(input_channel, output_channel)
        self.triplets = triplets
        self.img_dim_change = nn.Linear(512,64)
        self.txt_dim_change = nn.Linear(384,64)
        self.emb_size = config.dim
        self.regs = config.l2
        #self.regs = eval(args_config.regs)

        self.all_embed = self._init_weight(triplets)

    def _init_weight(self,triplets):
        all_embed = nn.Parameter(
            torch.FloatTensor(self.n_nodes, self.emb_size), requires_grad=True
        )
        ui = self.n_users + self.n_items

        if self.config.pretrain_r:
            nn.init.xavier_uniform_(all_embed)
            all_embed.data[:ui] = self.data_config["all_embed"]
        else:
            nn.init.xavier_uniform_(all_embed)

        img_embedding = pickle.load(open(self.config.img_embedding_path, 'rb'))
        txt_embedding = pickle.load(open(self.config.txt_embedding_path, 'rb'))
        img_embedding[1] = self.img_dim_change(img_embedding[1])
        txt_embedding[1] = self.txt_dim_change(txt_embedding[1])
        img_embedding[0] = [i + self.n_users for i in img_embedding[0]]
        txt_embedding[0] = [i + self.n_users for i in txt_embedding[0]]
        # img_embedding[1].requires_grad = True
        # txt_embedding[1].requires_grad = True

        comprise_dict = {}
        for i in triplets:
            if i[1] == 1:
                outfit = i[0]
                item = i[2]
                if outfit not in comprise_dict:
                    comprise_dict[outfit] = [item]
                else:
                    comprise_dict[outfit].append(item)

        if self.config.item:
            for num, i in enumerate(img_embedding[0]):
                with torch.no_grad():
                    all_embed[i] = 0.5 * (img_embedding[1][num] + txt_embedding[1][num])

        if self.config.outfit:
            for outfit in comprise_dict:
                items = comprise_dict[outfit]
                # length = len(items)
                items_embedding = []
                for i in items:
                    index = img_embedding[0].index(i)
                    items_embedding.append(0.5 * (img_embedding[1][index] + txt_embedding[1][index]))
                items_embedding = torch.stack(items_embedding)
                with torch.no_grad():
                    all_embed[outfit] = torch.mean(items_embedding, dim=0)

        if self.config.outfit_item:
            for num, i in enumerate(img_embedding[0]):
                with torch.no_grad():
                    all_embed[i] = 0.5 * (img_embedding[1][num] + txt_embedding[1][num])

            for outfit in comprise_dict:
                items = comprise_dict[outfit]
                items_embedding = []
                for i in items:
                    index = img_embedding[0].index(i)
                    items_embedding.append(0.5 * (img_embedding[1][index] + txt_embedding[1][index]))
                items_embedding = torch.stack(items_embedding)
                with torch.no_grad():
                    all_embed[outfit] = torch.mean(items_embedding, dim=0)

        return all_embed
    def build_edge(self, adj_matrix):
        """build edges based on adj_matrix"""
        sample_edge = self.config.edge_threshold
        edge_matrix = adj_matrix

        n_node = edge_matrix.size(0)
        node_index = (
            torch.arange(n_node, device=edge_matrix.device)
            .unsqueeze(1)
            .repeat(1, sample_edge)
            .flatten()
        )
        neighbor_index = edge_matrix.flatten()
        edges = torch.cat((node_index.unsqueeze(1), neighbor_index.unsqueeze(1)), dim=1)
        return edges

    def forward(self, data_batch, adj_matrix, edge_matrix):
    #def forward(self, data_batch, adj_matrix, edge_matrix):
        users = data_batch["users"]
        pos = data_batch["pos_items"]
        self.edges = self.build_edge(edge_matrix)

        neg_list = torch.tensor([], dtype=torch.long, device=adj_matrix.device)
        prob_list = torch.tensor([], device=adj_matrix.device)
        neg_list_user = torch.tensor([], dtype=torch.long, device=adj_matrix.device)
        prob_list_user = torch.tensor([], device=adj_matrix.device)

        k = self.config.k_step
        assert k > 0
        for _ in range(k):
            #OUTFIT 
            candidate_neg, two_hop_logits = self.kg_step_outfit(pos, users, adj_matrix, step=2)
            candidate_neg = self.filter_entity(candidate_neg, self.item_range)
            good_neg, good_logits = self.prune_step(
                self.dis, candidate_neg, users, two_hop_logits
            )
            good_logits = good_logits
            good_neg = good_neg

            neg_list = torch.cat([neg_list, good_neg.unsqueeze(0)])
            prob_list = torch.cat([prob_list, good_logits.unsqueeze(0)])
            
            #USER
            candidate_neg_user, one_hop_logits = self.kg_step_user(pos, users, adj_matrix, step=1)
            candidate_neg_user = self.filter_entity_user(candidate_neg_user, self.n_users)

            good_neg_user = candidate_neg_user
            good_logits_user = one_hop_logits

            neg_list_user = torch.cat([neg_list_user, good_neg_user.unsqueeze(0)])
            prob_list_user = torch.cat([prob_list_user, good_logits_user.unsqueeze(0)])

        return neg_list, prob_list, neg_list_user, prob_list_user
        

        '''
        for _ in range(k):
            """sample candidate negative items based on knowledge graph"""
            #one_hop, one_hop_logits = self.kg_step_outfit(pos, users, adj_matrix, step=1)

            one_hop_item, one_hop_logits = self.kg_step_outfit2item(pos, users, adj_matrix, step=1)
            
            """
            flag_one_hop_list = []
            count_0_step1 = 0 
            for temp_node in range(len(one_hop)):
                flag_one_hop = (one_hop[temp_node]>=114728 and one_hop[temp_node]<=139689).item()
                flag_one_hop_list.append(flag_one_hop)
                if flag_one_hop==0:
                    count_0_step1 += 1
            """

            candidate_neg, two_hop_logits = self.kg_step_outfit(
                one_hop_item, users, adj_matrix, step=2
            )
            
            """
            flag_two_hop_list = []
            ratio_list =  []
            for temp_node_indx in range(candidate_neg.shape[0]):
                count_0_step2 = 0
                temp_node_list = candidate_neg[temp_node_indx]
                for temp_node in range(len(temp_node_list)):
                    flag_two_hop = (temp_node_list[temp_node]>=114728 and temp_node_list[temp_node]<=139689).item()
                    flag_two_hop_list.append(flag_two_hop)
                    if flag_two_hop==0:
                        count_0_step2 += 1
                ratio_list.append((count_0_step2/len(temp_node_list)))
            pdb.set_trace()
            """

            candidate_neg = self.filter_entity(candidate_neg, self.item_range)
            good_neg, good_logits = self.prune_step(
                self.dis, candidate_neg, users, two_hop_logits
            )
            good_logits = good_logits + one_hop_logits

            neg_list = torch.cat([neg_list, good_neg.unsqueeze(0)])
            prob_list = torch.cat([prob_list, good_logits.unsqueeze(0)])

            pos = good_neg

        return neg_list, prob_list, neg_list, prob_list
        '''

        """
        one_hop, one_hop_logits = self.kg_step(pos, users, adj_matrix, step=1)

        candidate_neg, two_hop_logits = self.kg_step(
            one_hop, users, adj_matrix, step=2
        )
        candidate_neg = self.filter_entity(candidate_neg, self.item_range)
        good_neg, good_logits = self.prune_step(
            self.dis, candidate_neg, users, two_hop_logits
        )
        good_logits = good_logits + one_hop_logits

        neg_list = torch.cat([neg_list, good_neg.unsqueeze(0)])
        prob_list = torch.cat([prob_list, good_logits.unsqueeze(0)])
        pos = good_neg
        """
        

        '''
        candidate_neg, two_hop_logits = self.kg_step_outfit(pos, users, adj_matrix, step=2)
        candidate_neg = self.filter_entity(candidate_neg, self.item_range)
        good_neg, good_logits = self.prune_step(
            self.dis, candidate_neg, users, two_hop_logits
        )
        good_logits = good_logits
        good_neg = good_neg

        neg_list = torch.cat([neg_list, good_neg.unsqueeze(0)])
        prob_list = torch.cat([prob_list, good_logits.unsqueeze(0)])
        '''


        '''
        k = self.config.k_step
        assert k > 0
        for _ in range(k):
            #OUTFIT 
            candidate_neg, two_hop_logits = self.kg_step_outfit(pos, users, adj_matrix, step=2)
            candidate_neg = self.filter_entity(candidate_neg, self.item_range)
            good_neg, good_logits = self.prune_step(
                self.dis, candidate_neg, users, two_hop_logits
            )
            good_logits = good_logits
            good_neg = good_neg

            neg_list = torch.cat([neg_list, good_neg.unsqueeze(0)])
            prob_list = torch.cat([prob_list, good_logits.unsqueeze(0)])
            
            #USER
            candidate_neg_user, one_hop_logits = self.kg_step_user(pos, users, adj_matrix, step=1)
            candidate_neg_user = self.filter_entity_user(candidate_neg_user, self.n_users)

            good_neg_user = candidate_neg_user
            good_logits_user = one_hop_logits

            neg_list_user = torch.cat([neg_list_user, good_neg_user.unsqueeze(0)])
            prob_list_user = torch.cat([prob_list_user, good_logits_user.unsqueeze(0)])

        return neg_list, prob_list, neg_list_user, prob_list_user
        '''
    def kg_step_user(self, pos, user, adj_matrix, step):
        #x = self.entity_embedding
        edges = self.edges
        x = self.all_embed
        gcn_embedding = self.gcn(x, edges.t().contiguous())
        #gcn_embedding = torch.cat([user_gcn_emb,entity_gcn_emb], dim=0)

        """knowledge graph embedding using gcn"""
        #gcn_embedding = torch.cat([entity_gcn_emb,user_gcn_emb], dim=0)
        #gcn_embedding = self.gcn(x, edges.t().contiguous())

        """use knowledge embedding to decide candidate negative items"""
        u_e = gcn_embedding[user]
        u_e = u_e.unsqueeze(dim=1)
        pos_e = gcn_embedding[pos]
        pos_e = pos_e.unsqueeze(dim=2)

        one_hop_user = adj_matrix[user]
        i_user_e = gcn_embedding[one_hop_user]

        p_entity = F.leaky_relu(u_e * i_user_e)
        p = torch.sum(p_entity,dim=-1)
        '''
        p = torch.matmul(p_entity, pos_e)
        p = p.squeeze()
        '''
        logits = F.softmax(p, dim=1)

        """sample negative items based on logits"""
        batch_size = logits.size(0)
        if step == 1:
            nid = torch.argmax(logits, dim=1, keepdim=True)
        else:
            n = self.config.num_sample
            _, indices = torch.sort(logits, descending=True)
            nid = indices[:, :n]
        row_id = torch.arange(batch_size, device=logits.device).unsqueeze(1)

        candidate_neg = one_hop_user[row_id, nid].squeeze()
        candidate_logits = torch.log(logits[row_id, nid]).squeeze()

        return candidate_neg, candidate_logits


    def kg_step_outfit2item(self, pos, user, adj_matrix, step):
        #x = self.entity_embedding
        edges = self.edges
        x = self.all_embed
        gcn_embedding = self.gcn(x, edges.t().contiguous())
        #gcn_embedding = self.gcn(x, edges.t().contiguous())

        """use knowledge embedding to decide candidate negative items"""
        u_e = gcn_embedding[user]
        u_e = u_e.unsqueeze(dim=2)
        pos_e = gcn_embedding[pos]
        pos_e = pos_e.unsqueeze(dim=1)

        one_hop = adj_matrix[pos]
        #one_hop = self.filter_entity_item(one_hop, self.item_range)
        #139689-185310
        i_e = gcn_embedding[one_hop]

        p_entity = F.leaky_relu(pos_e * i_e)
        p = torch.sum(p_entity, dim=-1)
        '''
        p = torch.matmul(p_entity, u_e)
        p = p.squeeze()
        '''
        logits = F.softmax(p, dim=1)

        """sample negative items based on logits"""
        batch_size = logits.size(0)
        if step == 1:
            nid = torch.argmax(logits, dim=1, keepdim=True)
        else:
            n = self.config.num_sample
            _, indices = torch.sort(logits, descending=True)
            nid = indices[:, :n]
        row_id = torch.arange(batch_size, device=logits.device).unsqueeze(1)

        candidate_neg = one_hop[row_id, nid].squeeze()
        candidate_logits = torch.log(logits[row_id, nid]).squeeze()

        return candidate_neg, candidate_logits        

    def kg_step_outfit(self, pos, user, adj_matrix, step):
        #x = self.entity_embedding
        edges = self.edges
        x = self.all_embed
        gcn_embedding = self.gcn(x, edges.t().contiguous())
        #gcn_embedding = self.gcn(x, edges.t().contiguous())

        """use knowledge embedding to decide candidate negative items"""
        u_e = gcn_embedding[user]
        u_e = u_e.unsqueeze(dim=2)
        pos_e = gcn_embedding[pos]
        pos_e = pos_e.unsqueeze(dim=1)

        one_hop = adj_matrix[pos]
        i_e = gcn_embedding[one_hop]

        p_entity = F.leaky_relu(pos_e * i_e)
        p = torch.sum(p_entity, dim=-1)
        '''
        p = torch.matmul(p_entity, u_e)
        p = p.squeeze()
        '''
        logits = F.softmax(p, dim=1)

        """sample negative items based on logits"""
        batch_size = logits.size(0)
        if step == 1:
            nid = torch.argmax(logits, dim=1, keepdim=True)
        else:
            n = self.config.num_sample
            _, indices = torch.sort(logits, descending=True)
            nid = indices[:, :n]
        row_id = torch.arange(batch_size, device=logits.device).unsqueeze(1)

        candidate_neg = one_hop[row_id, nid].squeeze()
        candidate_logits = torch.log(logits[row_id, nid]).squeeze()

        return candidate_neg, candidate_logits

    def kg_step(self, pos, user, adj_matrix, step):
        #x = self.entity_embedding
        edges = self.edges
        x = self.all_embed
        gcn_embedding = self.gcn(x, edges.t().contiguous())
        """knowledge graph embedding using gcn"""
        #gcn_embedding = torch.cat([entity_gcn_emb,user_gcn_emb], dim=0)
        #gcn_embedding = self.gcn(x, edges.t().contiguous())

        """use knowledge embedding to decide candidate negative items"""
        u_e = gcn_embedding[user]
        u_e = u_e.unsqueeze(dim=2)
        pos_e = gcn_embedding[pos]
        pos_e = pos_e.unsqueeze(dim=1)

        one_hop = adj_matrix[pos]
        i_e = gcn_embedding[one_hop]

        p_entity = F.leaky_relu(pos_e * i_e)
        p = torch.matmul(p_entity, u_e)
        p = p.squeeze()
        logits = F.softmax(p, dim=1)

        """sample negative items based on logits"""
        batch_size = logits.size(0)
        if step == 1:
            nid = torch.argmax(logits, dim=1, keepdim=True)
        else:
            n = self.config.num_sample
            _, indices = torch.sort(logits, descending=True)
            nid = indices[:, :n]
        row_id = torch.arange(batch_size, device=logits.device).unsqueeze(1)

        candidate_neg = one_hop[row_id, nid].squeeze()
        candidate_logits = torch.log(logits[row_id, nid]).squeeze()

        return candidate_neg, candidate_logits

    def filter_entity_item(self, neg, item_range):
        low_range = int(item_range[1]+1)
        high_range = self.n_nodes
        random_neg = torch.randint(low_range, high_range, neg.size(), device=neg.device)

        neg[neg > high_range] = random_neg[neg > high_range]
        neg[neg < low_range] = random_neg[neg < low_range]

        '''
        random_neg = torch.randint(
            int(item_range[0]), int(item_range[1] + 1), neg.size(), device=neg.device
        )
        neg[neg > item_range[1]] = random_neg[neg > item_range[1]]
        neg[neg < item_range[0]] = random_neg[neg < item_range[0]]
        '''
        return neg

    @staticmethod
    def prune_step(dis, negs, users, logits):
        with torch.no_grad():
            #pdb.set_trace()
            ranking = dis.rank(users, negs)

        """get most qualified negative item based on user-neg similarity"""
        indices = torch.argmax(ranking, dim=1)

        batch_size = negs.size(0)
        row_id = torch.arange(batch_size, device=negs.device).unsqueeze(1)
        indices = indices.unsqueeze(1)

        good_neg = negs[row_id, indices].squeeze()
        goog_logits = logits[row_id, indices].squeeze()

        return good_neg, goog_logits

    @staticmethod
    def filter_entity(neg, item_range):
        random_neg = torch.randint(
            int(item_range[0]), int(item_range[1] + 1), neg.size(), device=neg.device
        )
        neg[neg > item_range[1]] = random_neg[neg > item_range[1]]
        neg[neg < item_range[0]] = random_neg[neg < item_range[0]]
        return neg

    @staticmethod
    def filter_entity_outfit(neg, item_range):
        random_neg = torch.randint(
            int(item_range[0]), int(item_range[1] + 1), neg.size(), device=neg.device
        )
        neg[neg > item_range[1]] = random_neg[neg > item_range[1]]
        neg[neg < item_range[0]] = random_neg[neg < item_range[0]]
        return neg

    @staticmethod
    def filter_entity_user(neg, n_users):
        random_neg = torch.randint(0, n_users, neg.size(), device=neg.device
        )
        neg[neg > n_users] = random_neg[neg > n_users]
        neg[neg < 0] = random_neg[neg < 0]
        return neg


