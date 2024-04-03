from arg_parser import parse_args
import random
import torch
import numpy as np
from data_loader import load_data, load_features
import networkx as nx
import time
from utils import *
from models import MM_model
import os
import pickle
from torch.utils.tensorboard import SummaryWriter




if __name__ == '__main__':
    """fix the random seed"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    writer = SummaryWriter('/root/tf-logs')
    st_time = time.time()

    """read args"""
    args = parse_args()
    device_cpu = torch.device("cpu")
    device_gpu = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")
    
    """load dataset"""
    train_data, test_data, outfit_item_data, test_users, train_users, train_outfits, train_items, user2outfit_train, user2outfit_gt, n_users, outfit2user_train = load_data(args)
    """build graph"""
    uoi_graph = build_graph(train_data, outfit_item_data)
    edge_list = [e for e in uoi_graph.edges]
    edge_indices = torch.tensor(edge_list).t().contiguous()
    """load feature"""
    entity_feat, image_feat, text_feat = load_features(args, uoi_graph.number_of_nodes(), n_users)


    """sample negatives"""
    print("{:.2f}: start sampling negatives".format(time.time()-st_time))
    if os.path.exists("cf_negs_uoo.pickle"): 
        f=open('cf_negs_uoo.pickle','rb')
        cf_negs_uoo=pickle.load(f)
        f.close()
        f=open('cf_negs_ouu.pickle','rb')
        cf_negs_ouu=pickle.load(f)
        f.close()
        f=open('cl_negs_user.pickle','rb')
        cl_negs_user=pickle.load(f)
        f.close()
        f=open('cl_negs_outfit.pickle','rb')
        cl_negs_outfit=pickle.load(f)
        f.close()
        f=open('cl_negs_item.pickle','rb')
        cl_negs_item=pickle.load(f)
        f.close()
    else:
        cf_negs_uoo, cf_negs_ouu, cl_negs_user, cl_negs_outfit, cl_negs_item = sample_negatives(uoi_graph, train_users, train_outfits, train_items, args.num_negs, user2outfit_train, outfit2user_train)
        f=open('cf_negs_uoo.pickle','wb')
        pickle.dump(cf_negs_uoo,f)
        f.close()
        f=open('cf_negs_ouu.pickle','wb')
        pickle.dump(cf_negs_ouu,f)
        f.close()
        f=open('cl_negs_user.pickle','wb')
        pickle.dump(cl_negs_user,f)
        f.close()
        f=open('cl_negs_outfit.pickle','wb')
        pickle.dump(cl_negs_outfit,f)
        f.close()
        f=open('cl_negs_item.pickle','wb')
        pickle.dump(cl_negs_item,f)
        f.close()
    print("{:.2f}: finish sampling negatives".format(time.time()-st_time))

    model = MM_model(args, entity_feat.shape[1], image_feat.shape[1], text_feat.shape[1], uoi_graph.number_of_nodes())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    """load model and data to GPU"""
    model = model.to(device_gpu)
    entity_feat = entity_feat.to(device_gpu)
    image_feat = image_feat.to(device_gpu)
    text_feat = text_feat.to(device_gpu)
    edge_indices = edge_indices.to(device_gpu)

    for epoch in range(args.epoch):
        model.train()
        print("{:.2f}: start ephoch {}".format(time.time()-st_time, epoch))
        epoch_loss = 0
        epoch_bpr_loss = 0
        epoch_reg_loss = 0
        epoch_cl_loss = 0
        batch_num = 0

        # shuffle train dataset
        index = np.arange(len(train_data))
        np.random.shuffle(index)
        train_data = train_data[index]

        while (batch_num+1)*args.batch_size < len(train_data):
            batch_loss = 0
            entity_emb, image_emb, text_emb, mm_emb = model(entity_feat, image_feat, text_feat, edge_indices)
            batch_data = train_data[batch_num*args.batch_size:(batch_num+1)*args.batch_size]
            user_ids = batch_data[:,0]
            item_ids = batch_data[:,1]

            # cal bpr loss
            # (user, outfit+, outfit-)
            outfits_neg = []
            for i, u in enumerate(user_ids):
                outfits_neg.append(cf_negs_uoo[u][(i+batch_num*args.batch_size)%1000])     # 22307 is a prime number and less than num_users(24961)
                # outfits_neg.append(random.sample(train_outfits, 1)[0])

            user_emb = mm_emb[user_ids]
            outfit_emb_pos = mm_emb[item_ids]
            outfit_emb_neg = mm_emb[outfits_neg]
            
            pos_scores_uoo = torch.sum(user_emb * outfit_emb_pos, dim=1)
            neg_scores_uoo = torch.sum(user_emb * outfit_emb_neg, dim=1)
            bpr_loss_uoo = torch.log(torch.sigmoid(pos_scores_uoo - neg_scores_uoo))
            bpr_loss_uoo = -torch.mean(bpr_loss_uoo)

            # (outfit, user+, user-)
            users_neg = []
            for i, o in enumerate(item_ids):
                users_neg.append(cf_negs_ouu[o][(i+batch_num*args.batch_size)%4000])     # 22307 is a prime number and less than num_users(24961)
                # outfits_neg.append(random.sample(train_outfits, 1)[0])

            outfit_emb = mm_emb[item_ids]
            user_emb_pos = mm_emb[user_ids]
            user_emb_neg = mm_emb[users_neg]
            
            pos_scores_ouu = torch.sum(user_emb * outfit_emb_pos, dim=1)
            neg_scores_ouu = torch.sum(user_emb * outfit_emb_neg, dim=1)
            bpr_loss_ouu = torch.log(torch.sigmoid(pos_scores_ouu - neg_scores_ouu))
            bpr_loss_ouu = -torch.mean(bpr_loss_ouu)
            
            bpr_loss = bpr_loss_uoo + bpr_loss_ouu
            # bpr_loss = bpr_loss_uoo


            # cal reg loss
            reg_loss = torch.sum(user_emb ** 2) / 2 + torch.sum(outfit_emb_pos ** 2) / 2 + torch.sum(outfit_emb_neg ** 2) / 2

            # # cal cl loss for user nodes
            # cl_loss_user_entity_img = loss_cl(entity_emb[user_ids], image_emb[user_ids], args.tau)
            # cl_loss_user_entity_txt = loss_cl(entity_emb[user_ids], text_emb[user_ids], args.tau)
            # cl_loss_user_img_txt = loss_cl(image_emb[user_ids], text_emb[user_ids], args.tau)

            # # cal cl loss for item nodes
            # cl_loss_item_entity_img = loss_cl(entity_emb[item_ids], image_emb[item_ids], args.tau)
            # cl_loss_item_entity_txt = loss_cl(entity_emb[item_ids], text_emb[item_ids], args.tau)
            # cl_loss_item_img_txt = loss_cl(image_emb[item_ids], text_emb[item_ids], args.tau)


            # cl_loss_user = (cl_loss_user_entity_img+cl_loss_user_entity_txt+cl_loss_user_img_txt) / 3
            # cl_loss_item = (cl_loss_item_entity_img+cl_loss_item_entity_txt+cl_loss_item_img_txt) / 3
            # cl_loss = cl_loss_user + cl_loss_item

            # total loss
            # batch_loss = bpr_loss + args.l2 * reg_loss + args.wcl * cl_loss
            batch_loss = bpr_loss + args.l2 * reg_loss
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += batch_loss.item()  # if without .item(), this will cause ram leakage
            epoch_bpr_loss += bpr_loss.item()
            epoch_reg_loss += args.l2 * reg_loss.item()
            # epoch_cl_loss += args.wcl * cl_loss.item()
            batch_num += 1
            # if batch_num%200 == 0:
            #     print("{:.2f}: finish batch {}/{}. loss: {:.3f}. bpr loss: {}; reg_loss: {}".format(time.time()-st_time, batch_num, int(len(train_data)/args.batch_size), 
            #                                                                                                      batch_loss, bpr_loss, args.l2 * reg_loss))
        
        # print("{:.2f}: train loss: {:.3f}. bpr loss: {}; cl_loss: {}".format(time.time()-st_time, epoch_loss, epoch_bpr_loss, epoch_cl_loss))
        print("{:.2f}: train loss: {:.3f}. bpr loss: {}. reg loss: {}".format(time.time()-st_time, epoch_loss, epoch_bpr_loss, epoch_reg_loss))
        writer.add_scalar("Train/train_loss", epoch_loss, epoch)
        writer.add_scalar("Train/bpr_loss", epoch_bpr_loss, epoch)
        writer.add_scalar("Train/reg_loss", epoch_reg_loss, epoch)

        if epoch % args.test_step==0:
            model.eval()
            with torch.no_grad():
                entity_emb, image_emb, text_emb, mm_emb = model(entity_feat, image_feat, text_feat, edge_indices)
                test_users_list = list(test_users)
                train_outfits_list = list(train_outfits)
                test_user_emb = mm_emb[test_users_list]
                outfit_emb = mm_emb[train_outfits_list]
                score_matrix = test_user_emb @ outfit_emb.T
                ks = [20]   # K's in Top K, e.g. [20, 50, 100] means test with top10, top 20,..., top100
                precision_list = []
                recall_list = []
                ndcg_list = []
                hr_list = []
                
                for k in ks:
                    _, indices = torch.topk(score_matrix , k, dim=1)
                    precision, recall, ndcg, hr = 0, 0, 0, 0

                    for i in range(indices.shape[0]):
                        u_id = test_users_list[i]
                        o_ids_recommended  = torch.Tensor(train_outfits_list)[indices[i]].int().tolist()
                        o_ids_gt = user2outfit_gt[u_id]
                        num_overlap = len(set(o_ids_recommended) & set(o_ids_gt))
                        precision += num_overlap/len(o_ids_recommended)
                        recall += num_overlap/len(o_ids_gt)
                        if num_overlap > 0:
                            hr += 1
                        ndcg += cal_ndcg(o_ids_recommended, o_ids_gt)


                    precision_list.append(precision/indices.shape[0])
                    recall_list.append(recall/indices.shape[0])
                    ndcg_list.append(ndcg/indices.shape[0])
                    hr_list.append(hr/indices.shape[0])
                del score_matrix
                torch.cuda.empty_cache()
                print("{:.2f}: Test Top 20: precision {}, recall {}, ndcg {}, hr {}".format(time.time()-st_time, precision_list[0], recall_list[0], ndcg_list[0], hr_list[0]))
                # print("{:.2f}: Top40 precision {}, recall {}, ndcg {}, hr {}".format(time.time()-st_time, precision_list[1], recall_list[1], ndcg_list[1], hr_list[1]))
                # print("{:.2f}: Top100 precision {}, recall {}, ndcg {}, hr {}".format(time.time()-st_time, precision_list[2], recall_list[2], ndcg_list[2], hr_list[2]))
                writer.add_scalar("Test/precision", precision_list[0], epoch)
                writer.add_scalar("Test/recall", recall_list[0], epoch)
                writer.add_scalar("Test/ndcg", ndcg_list[0], epoch)
                writer.add_scalar("Test/hr", hr_list[0], epoch)

                