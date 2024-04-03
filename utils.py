from arg_parser import parse_args
import random
import torch
import numpy as np
from data_loader import load_data, load_features
import networkx as nx
import time


def build_graph(train_data, outfit_item_data):
    # return:
    # n_users: total number of users, also the offset of outfit id and item id
    # uoi_graph: user_outfit_item graph, without node feature

    uoi_graph = nx.DiGraph()

    max_id = max(max(outfit_item_data[:, 0]), max(outfit_item_data[:, 1]))
    node_list = [i for i in range(max_id+1)]
    uoi_graph.add_nodes_from(node_list)

    for pair in train_data:
        u_id = pair[0]
        o_id = pair[1]
        uoi_graph.add_edge(o_id, u_id)
    for pair in outfit_item_data:
        o_id = pair[0]
        i_id = pair[1]
        uoi_graph.add_edge(i_id, o_id)

    return uoi_graph


def sample_negatives(graph, users, outfits, items, num_negs, user2outfit_train, outfit2user_train):
    """
    return:
    cf_negs_uoo: negative outfits for (user, outfit, outfit-) in BPR loss, dict: int->list
    cf_negs_ouu: negative user for (user, outfit, user-) in BPR loss, dict: int->list
    cl_negs_user: negative users for each user in CL loss, dict
    cl_negs_outfit: negative outfits for each outfit in CL loss, dict
    cl_negs_item: negative items for each item in CL loss, dict
    """
    cf_negs_uoo = {}
    count = 0
    for u in users:
        interacted_outfits = set(user2outfit_train[u])
        uninteracted_outfits = outfits - interacted_outfits
        cf_negs_uoo[u] = random.sample(uninteracted_outfits, num_negs)
        # cf_negs_uoo[u] = list(uninteracted_outfits)
        if count%1000==0:
            print("{}/{} cf (u,o+,o-) samples done".format(count, len(users)))
        count += 1

    cf_negs_ouu = {}
    count = 0
    for o in outfits:
        interacted_users = set(outfit2user_train[o])
        uninteracted_users = users - interacted_users
        cf_negs_ouu[o] = random.sample(uninteracted_users, 4*num_negs)
        # cf_negs_uoo[u] = list(uninteracted_outfits)
        if count%1000==0:
            print("{}/{} cf (o,u+,u-) samples done".format(count, len(outfits)))
        count += 1
        
    cl_negs_user = {}
    # count = 0
    # for u in users:
    #     tempset = set()
    #     tempset.add(u)
    #     cl_negs_user[u] = random.sample(users - tempset, num_negs)
    #     if count%1000==0:
    #         print("{}/{} cl user samples done".format(count, len(users)))
    #     count += 1
    
    cl_negs_outfit = {}
    # count = 0
    # for o in outfits:
    #     tempset = set()
    #     tempset.add(o)
    #     cl_negs_outfit[o] = random.sample(outfits - tempset, num_negs)
    #     if count%1000==0:
    #         print("{}/{} cl outfit samples done".format(count, len(outfits)))
    #     count += 1
    
    cl_negs_item = {}
    # count = 0
    # for i in items:
    #     tempset = set()
    #     tempset.add(i)
    #     cl_negs_item[i] = random.sample(items - tempset, num_negs)
    #     if count%1000==0:
    #         print("{}/{} cl item samples done".format(count, len(items)))
    #     count += 1
        
    return cf_negs_uoo, cf_negs_ouu, cl_negs_user, cl_negs_outfit, cl_negs_item


# def cal_cl_loss(anchor_emb_m1, anchor_emb_m2, neg_embs_m2):
#     '''
#     contrast modal 1 to modal 2 for an anchor node
#     anchor_emb_m1: (feat_dim)   embedding of anchor node in modal 1
#     anchor_emb_m2: (feat_dim)   embedding of anchor node in modal 2
#     neg_embs_m2: (num_negs, feat_dim)  embedding of neg nodes in modal 2
#     '''
#     up = torch.exp(torch.dot(anchor_emb_m1, anchor_emb_m2))
#     temp = anchor_emb_m1.expand(neg_embs_m2.shape[0], -1)
#     down = torch.sum(temp * neg_embs_m2, dim=1)
#     down = torch.sum(torch.exp(down))

#     ret = -torch.log(up / down)

#     return ret


def loss_cl(x1, x2, tau):
    '''
    contrast modal 1 to modal 2 for all nodes in a batch, using InfoNCE loss
    x1: (num_nodes_batch, feat_dim)   nodes' embedding in modal 1
    x2: (num_nodes_batch, feat_dim)   nodes' embedding in modal 2
    tau:  hyperparameter
    '''
    T = tau
    batch_size, _ = x1.size()
    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)

    # modal 1 -> modal 2
    sim_matrix_a = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
    sim_matrix_a = torch.exp(sim_matrix_a / T)
    pos_sim_a = sim_matrix_a[range(batch_size), range(batch_size)]
    loss_a = pos_sim_a / (sim_matrix_a.sum(dim=1) - pos_sim_a)
    loss_a = - torch.log(loss_a).mean()

    # modal 2 -> modal 1
    sim_matrix_b = torch.einsum('ik,jk->ij', x2, x1) / torch.einsum('i,j->ij', x2_abs, x1_abs)
    sim_matrix_b = torch.exp(sim_matrix_b / T)
    pos_sim_b = sim_matrix_b[range(batch_size), range(batch_size)]
    loss_b = pos_sim_b / (sim_matrix_b.sum(dim=1) - pos_sim_b)
    loss_b = - torch.log(loss_b).mean()

    loss = (loss_a + loss_b) / 2
    return loss


def cal_ndcg(recommended_list, gt_list):
    '''
    recommended_list:  outfit id list recommended by model for a user
    gt_list: ground truth outfit id list of a user (interacted in test data)
    '''
    label = [0 for i in range(len(recommended_list))]
    for i, o in enumerate(recommended_list):
        if o in gt_list:
            label[i] = 1
    label = np.array(label)
    dcg = np.sum(label / np.log(np.arange(2, len(label) + 2)))
    idcg = np.sum(sorted(label, reverse=True) / np.log(np.arange(2, len(label) + 2)))
    ndcg = dcg / idcg if idcg != 0 else 0

    return ndcg