import numpy as np
import torch
import pickle
import os

def read_outfit_item_data(file_name):
    # return [[oid,iid],...,[oid,iid]]
    triplets_np = np.loadtxt(file_name, dtype=np.int32)
    triplets_np = np.unique(triplets_np, axis=0)

    index = (triplets_np[:, 1] == 0)
    outfit_id_list = triplets_np[index, 0]
    item_id_list = triplets_np[index, 2]
    outfit_item_list = np.stack((outfit_id_list, item_id_list), axis=1)
    
    return outfit_item_list


def read_user_outfit_data(file_name):
    # return [[uid,oid],...,[uid,oid]]
    triplets = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        if len(l.strip().split('\t')) !=2:
            print('error')
            print(l)
        items = l.strip().split('\t')[0]
        rules = l.strip().split('\t')[1]
        inters = [int(i) for i in items.split(" ")]
        inters_rule = [int(i) for i in rules.split(" ")]
        u_id, pos_ids = inters[0], inters[1:]
        for index, o_id in enumerate(pos_ids):
            triplets.append([u_id, o_id, inters_rule[index]])

    # user-outfit-rule triple data        
    triplets=np.array(triplets)
    # user-outfit pair data
    user_outfit_list = triplets[:,0:2]

    return user_outfit_list


def load_data(args):
    # return:
    # train_data/test_data: [[uid,oid],...,[uid,oid]]
    # outfit_item_data: [[oid,iid],...,[oid,iid]]
    # test_users: all users in test data, set
    # train_users, train_outfits, train_items. as above. set
    # user2outfit_train: outfits that a user has interacted with in train data, dict: int->list
    # user2outfit_gt: outfits that a user has interacted with in test data (groundtruth), dict: int->list
    print('reading train and test user-item set ...')
    directory = args.data_path + args.dataset + '/'
    train_data = read_user_outfit_data(directory + 'train_remap_reorg_with_rule_June.txt')
    test_data = read_user_outfit_data(directory + 'test_remap_reorg_with_rule_June.txt')
    outfit_item_data = read_outfit_item_data(directory + 'kg_final_top_bottom_remap.txt')

    # add offset to outfit id and item id
    min_id = min(min(train_data[:, 0]), min(test_data[:, 0]))
    max_id = max(max(train_data[:, 0]), max(test_data[:, 0]))
    n_users = max_id - min_id + 1
    train_data[:, 1] = train_data[:, 1] + n_users
    test_data[:, 1] = test_data[:, 1] + n_users
    outfit_item_data[:, 0] = outfit_item_data[:, 0] + n_users
    outfit_item_data[:, 1] = outfit_item_data[:, 1] + n_users


    test_users = set(test_data[:,0])
    train_users = set(train_data[:,0])
    train_outfits = set(train_data[:,1])
    train_items = set()
    for pair in outfit_item_data:
        o_id = pair[0]
        i_id = pair[1]
        if o_id in train_outfits:
            train_items.add(i_id)

    user2outfit_train = {}
    outfit2user_train = {}
    for pair in train_data:
        uid = pair[0]
        oid = pair[1]
        if uid not in user2outfit_train.keys():
            user2outfit_train[uid] = []
            user2outfit_train[uid].append(oid)
        else:
            user2outfit_train[uid].append(oid)

        if oid not in outfit2user_train.keys():
            outfit2user_train[oid] = []
            outfit2user_train[oid].append(uid)
        else:
            outfit2user_train[oid].append(uid)


    user2outfit_gt = {}
    for pair in test_data:
        uid = pair[0]
        oid = pair[1]
        if uid not in user2outfit_gt.keys():
            user2outfit_gt[uid] = []
            user2outfit_gt[uid].append(oid)
        else:
            user2outfit_gt[uid].append(oid)

    return train_data, test_data, outfit_item_data, test_users, train_users, train_outfits, train_items, user2outfit_train, user2outfit_gt, n_users, outfit2user_train


def load_features(args, num_graph_nodes, offset):
    # return:
    # entity_feat: (num_graph_nodes, dim_ent)
    # image_feat: (num_graph_nodes, dim_img)
    # text_feat: (num_graph_nodes, dim_txt)

    dim_ent = args.dim

    if os.path.exists("ent_embedding.pickle"): 
        f=open('ent_embedding.pickle','rb')
        entity_feat=pickle.load(f)
        f.close()
    else:
        f=open('ent_embedding.pickle','wb')
        entity_feat = torch.rand(num_graph_nodes, dim_ent)
        pickle.dump(entity_feat,f)
        f.close()

    # entity_feat = torch.rand(num_graph_nodes, dim_ent)
    
    img_data = pickle.load(open(args.img_embedding_path, 'rb'))
    # add offset
    for i,k in enumerate(img_data[0]):
        img_data[0][i] = k + offset
    dim_img = img_data[1].shape[1]
    image_feat = torch.rand(num_graph_nodes, dim_img)
    for i,k in enumerate(img_data[0]):
        image_feat[k] = img_data[1][i]

    txt_data = pickle.load(open(args.txt_embedding_path, 'rb'))
    # add offset
    for i,k in enumerate(txt_data[0]):
        txt_data[0][i] = k + offset
    dim_txt = txt_data[1].shape[1]
    text_feat = torch.rand(num_graph_nodes, dim_txt)
    for i,k in enumerate(txt_data[0]):
        text_feat[k] = txt_data[1][i]

    return entity_feat, image_feat, text_feat