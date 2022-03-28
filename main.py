import time
import argparse

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import random

from utils.pytorchtools import EarlyStopping
from utils.data import load_DBLP_data
from utils.tools import index_generator, evaluate_results_nc, parse_minibatch_DBLP, get_aa_negtive_nodes, get_ap_negtive_nodes
from model.DHGNN_lp import *

import warnings
warnings.filterwarnings("ignore")

# Params
out_dim = 4
# dropout_rate = 0.2
# lr = 1e-4  
weight_decay = 0.001
etypes_list = [[0, 1], [0, 2, 3, 1], [0, 4, 5, 1]]
text_mask = [[1], [1, 3], [1, 3]]
use_masks = [True, True, True]  # 两边的元路径相同
no_masks = [False] * 3
# 0 represents paper-author
# 1 represents author-paper
# 2 represents paper-conference
# 3 represents conference-paper
# 4 represents paper-term
# 5 represents term-paper
def run_model_DBLP(feats_type, hidden_dim, asp_dim, num_heads, attn_vec_dim, rnn_type,
                   num_epochs, patience, batch_size, neighbor_samples, neg_num, repeat, save_postfix, gpu_num, seed, lr, max_iter, dropout_rate,lamda,ga,pr,prc):

    adjlists_aa, edge_metapath_indices_list_aa, features_list, topic_array, adjM, type_mask, train_val_test_pos_a_p, train_val_test_neg_a_p, adj_lists = load_DBLP_data(num_heads)
    device = torch.device('cuda:' + str(gpu_num) if torch.cuda.is_available() else 'cpu')
    features_list = [torch.FloatTensor(features).to(device) for features in features_list]
    topic = torch.FloatTensor(np.vstack([topic_array, np.array([[1/num_heads] * topic_array.shape[1]])])).to(device)
    print(topic_array.shape)

    if feats_type == 0:  # 不同feature不同维度列表
        in_dims = [features.shape[1] for features in features_list]

    elif feats_type == 1:
        in_dims = [features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(1, len(features_list)):
            features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2:
        in_dims = [features.shape[0] for features in features_list]
        in_dims[0] = features_list[0].shape[1]
        for i in range(1, len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)

    # 链接预测实验
    train_pos_a_p = train_val_test_pos_a_p['a_p_train_pos_candidates']

    val_pos_a_p = train_val_test_pos_a_p['a_p_test_pos_candidates']

    test_pos_a_p = train_val_test_pos_a_p['a_p_test_pos_candidates']
    train_a_nodes = list(train_val_test_neg_a_p['a_nodes'])
    train_p_nodes = list(train_val_test_neg_a_p['p_nodes'])
    aa_lists = adj_lists[0]
    ap_lists = adj_lists[1]

    val_neg_a_p = train_val_test_neg_a_p['a_p_test_neg_candidates']

    test_neg_a_p = train_val_test_neg_a_p['a_p_test_neg_candidates']
    y_true_test_p = np.array([1] * len(test_pos_a_p) + [0] * len(test_neg_a_p))

    # core
    train_core_align_label = torch.from_numpy(np.arange(0, num_heads)).to(device)
    auc_list = []
    ap_list = []
    val_auc_list = []
    val_ap_list = []
    with open('result_{}.txt'.format(save_postfix), 'w', encoding='utf8') as result_file:
        for _ in range(repeat):
            net = DHGNN_lp(  # 两种节点元路径类别，边类型个数，边类型的list, feauture的维度，
                3, 6, etypes_list, in_dims, asp_dim, hidden_dim, num_heads, attn_vec_dim, max_iter, rnn_type, dropout_rate)
            net.to(device)
            fn_loss = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

            # training loop
            net.train()  # 训练
            early_stopping = EarlyStopping(patience=patience, verbose=True, save_path='v1_checkpoint/checkpoint_{}.pt'.format(save_postfix))
            dur1 = []
            dur2 = []
            dur3 = []
            current_loss = 10000.0
            current_auc = 0.0
            train_pos_idx_generator = index_generator(batch_size=batch_size, num_data=len(train_pos_a_p))
            val_idx_generator_ap = index_generator(batch_size=batch_size, num_data=len(val_pos_a_p), shuffle=False)
            for epoch in range(num_epochs):
                t_start = time.time()
                # training
                net.train()

                for iteration in range(train_pos_idx_generator.num_iterations()):
                    # forward
                    t0 = time.time()

                    train_pos_idx_batch = train_pos_idx_generator.next()  # batch里的索引
                    train_pos_idx_batch.sort()
                    # [8,2]
                    train_pos_a_a_batch = train_pos_a_p[train_pos_idx_batch].tolist()  
                    train_pos_a_p_batch = list(map(lambda sample: [sample[0], sample[1]], train_pos_a_a_batch))

                    train_neg_a_p_batch = get_ap_negtive_nodes(train_p_nodes, ap_lists, [row[0] for row in train_pos_a_a_batch], neg_num)

                    train_pos_g_lists, train_pos_indices_lists, train_pos_text_indices_lists, train_pos_idx_batch_mapped_lists, train_pos_nodes_lists = parse_minibatch_DBLP(
                        adjlists_aa, edge_metapath_indices_list_aa, train_pos_a_a_batch, text_mask, topic_array, device,
                        a_p_batch=train_pos_a_p_batch, samples=neighbor_samples, use_masks=use_masks, modes=1)
                    train_neg_g_lists, train_neg_indices_lists, train_neg_text_indices_lists, train_neg_idx_batch_mapped_lists, train_neg_nodes_lists = parse_minibatch_DBLP(
                        adjlists_aa, edge_metapath_indices_list_aa, train_pos_a_a_batch, text_mask, topic_array, device,
                        a_p_batch=train_pos_a_p_batch, samples=neighbor_samples, use_masks=no_masks, modes=1)
                    # [neg_num, 3]


                    t1 = time.time()
                    dur1.append(t1 - t0)

                    a_pos_embedding_list, a_logits_pos_embedding_list, asp_features, align_scores, att_pos = net(
                        ([train_pos_g_lists], features_list, topic, [train_pos_text_indices_lists], type_mask,
                         [train_pos_indices_lists], [train_pos_idx_batch_mapped_lists], [train_pos_nodes_lists]))
                    a_neg_embedding_list, a_logits_neg_embedding_list, asp_features, align_scores_neg, att_neg = net(
                        ([train_neg_g_lists], features_list, topic, [train_neg_text_indices_lists], type_mask,
                         [train_neg_indices_lists], [train_neg_idx_batch_mapped_lists], [train_neg_nodes_lists]))

                    pos_embedding_a0 = F.elu(a_pos_embedding_list[0].view(-1, 1, asp_dim))  # [8k,1,64]
                    neg_embedding_a0 = F.elu(a_neg_embedding_list[0].view(-1, 1, asp_dim))


                    ''' 去掉relu'''
                    p_pos_embedding = F.elu(asp_features[np.array([row[1] for row in train_pos_a_a_batch])].view(-1, asp_dim, 1))  # [8, dim, 1]
                    p_neg_embedding = F.elu(torch.cat([asp_features[np.array(train_neg_a_p_batch)[:, i]].view(-1, asp_dim, 1)
                                                 for i in range(neg_num)], dim=-1)) # [8,dim,10]
                    # prior build 6 matrix
                    prior_list = [[], []]  # 正样本 和 负样本
                    prior_loss = 0.0
                    if pr == 0:
                        for mode, text in enumerate(train_pos_text_indices_lists):  # 正样本 目前单边是一个

                            for path in text:  # 3条元路径
                                prior_M = torch.mean(topic[path], 0)
                                prior_list[0].append(prior_M.squeeze(dim=0))  # [2,3]
                        for mode, text in enumerate(train_neg_text_indices_lists):  # 负样本 目前单边是一个

                            for path in text:
                                prior_M = torch.mean(topic[path], 0)
                                prior_list[1].append(prior_M.squeeze(dim=0))
                        for mode in range(1):
                            for posterior, prior in zip(att_pos[mode], prior_list[0]):  # 三条元路径
                                Pr = torch.diag(prior)
                                posterior = posterior.squeeze(dim=-1)
                                PP = posterior.t().mm(posterior)
                                prior_loss += torch.norm(PP / (torch.norm(PP) + 1e-9) - Pr / (torch.norm(Pr) + 1e-9))
                        for mode in range(1):
                            for posterior, prior in zip(att_neg[mode], prior_list[1]):  #
                                Pr = torch.diag(prior)
                                posterior = posterior.squeeze(dim=-1)
                                PP = posterior.t().mm(posterior)
                                prior_loss += torch.norm(PP / (torch.norm(PP) + 1e-9) - Pr / (torch.norm(Pr) + 1e-9))
                    elif pr == 1:
                        for mode, text in enumerate(train_pos_text_indices_lists):  # 正样本
                            for path in text:  # 3条元路径
                                prior_M = topic[path].squeeze(dim=1)  # m条实例
                                prior_list[0].append(prior_M)  # [2,3]
                        for mode, text in enumerate(train_neg_text_indices_lists):  # 负样本

                            for path in text:
                                prior_M = topic[path].squeeze(dim=1) # m条实例
                                prior_list[1].append(prior_M)
                        for mode in range(1):
                            for posterior, prior in zip(att_pos[mode], prior_list[0]):  # 三条元路径
                                prior_loss += F.kl_div(posterior.squeeze(dim=-1).log(), prior,reduction='batchmean') 
                        for mode in range(1):
                            for posterior, prior in zip(att_neg[mode], prior_list[1]):  #
                                prior_loss += F.kl_div(posterior.squeeze(dim=-1).log(), prior, reduction='batchmean')
                    else:
                        prior_loss += 0.0
                    if prc == 'mean':
                        prior_loss= prior_loss/(3+1e-9)  
                    else:
                        prior_loss = prior_loss



                    # 计算loss
                    pos_ap_out = torch.bmm(pos_embedding_a0, p_pos_embedding).view(-1, num_heads, 1).sum(dim=1)
                    neg_ap_out = torch.bmm(neg_embedding_a0, p_neg_embedding).view(-1, num_heads, neg_num).sum(dim=1)
                    regular_loss = F.cross_entropy(align_scores[0], train_core_align_label) + F.cross_entropy(align_scores_neg[0],train_core_align_label)
                    # prior_loss =
                    label_pos = torch.unsqueeze(torch.ones(pos_ap_out.shape[0]), -1).to(device)
                    label_neg = torch.unsqueeze(torch.zeros(pos_ap_out.shape[0]), -1).to(device)
                    train_loss = fn_loss(pos_ap_out, label_pos) + fn_loss(neg_ap_out, label_neg) + lamda * regular_loss+ ga *prior_loss
                    # print(torch.max(torch.bmm(pos_embedding_a0, pos_embedding_a1).view(-1, num_heads, 1), 1)[1])
                    t2 = time.time()
                    dur2.append(t2 - t1)

                    # autograd
                    optimizer.zero_grad()  
                    train_loss.backward() 
                    optimizer.step() 

                    t3 = time.time()
                    dur3.append(t3 - t2)

                    # print training info
                    if iteration % 800 == 0:

                        print(
                            'Epoch {:05d} | Iteration {:05d} | Train_Loss {:.4f} |topic_Loss {:.4f} |ga_topic_Loss {:.4f}| Time1(s) {:.4f} | Time2(s) {:.4f} | Time3(s) {:.4f}'.format(
                                epoch, iteration, train_loss.item(), prior_loss.item(), (ga*prior_loss).item(), np.mean(dur1), np.mean(dur2), np.mean(dur3)))

                # validation
                net.eval()
                val_loss = []
                ap_val_loss = []


                ap_pos_proba_list = []
                ap_neg_proba_list = []

                pos_att_list_a0_0 = []
                pos_att_list_a0_1 = []
                pos_att_list_a0_2 = []
                neg_att_list_a0_0 = []
                neg_att_list_a0_1 = []
                neg_att_list_a0_2 = []

                a_val_pos_embedding = []
                a_val_neg_embedding = []
                p_val_pos_embedding = []
                p_val_neg_embedding = []
                pos_max = []

                with torch.no_grad():  # 没有梯度

                    for iteration in range(val_idx_generator_ap.num_iterations()):
                        # forward
                        val_idx_batch = val_idx_generator_ap.next()
                        val_pos_a_p_batch = val_pos_a_p[val_idx_batch].tolist()
                        val_neg_a_p_batch = val_neg_a_p[val_idx_batch].tolist()


                        val_pos_g_lists, val_pos_indices_lists, val_pos_text_indices_lists, val_pos_idx_batch_mapped_lists, val_pos_nodes_lists = parse_minibatch_DBLP(
                            adjlists_aa, edge_metapath_indices_list_aa, val_pos_a_p_batch, text_mask, topic_array, device, samples=neighbor_samples, use_masks=no_masks, modes=1)
                        val_neg_g_lists, val_neg_indices_lists, val_neg_text_indices_lists, val_neg_idx_batch_mapped_lists, val_neg_nodes_lists = parse_minibatch_DBLP(
                            adjlists_aa, edge_metapath_indices_list_aa, val_neg_a_p_batch, text_mask, topic_array, device, samples=neighbor_samples, use_masks=no_masks, modes=1)

                        a_pos_embedding_list, a_logits_pos_embedding_list, asp_features, pos_align_scores, att_pos= net(
                            ([val_pos_g_lists], features_list, topic, [val_pos_text_indices_lists], type_mask,
                             [val_pos_indices_lists], [val_pos_idx_batch_mapped_lists], [val_pos_nodes_lists]))
                        a_neg_embedding_list, a_logits_neg_embedding_list, asp_features, neg_align_scores, att_neg = net(
                            ([val_neg_g_lists], features_list, topic,
                             [val_neg_text_indices_lists], type_mask,
                             [val_neg_indices_lists],
                             [val_neg_idx_batch_mapped_lists],
                             [val_neg_nodes_lists]))

                        # a_neg_embedding_list, _ = net(
                        #     (val_neg_g_lists, features_list, topic, val_neg_text_indices_lists, type_mask, val_neg_indices_lists, val_neg_idx_batch_mapped_lists, val_neg_nodes_lists, True))

                        p_pos_embedding = F.elu(asp_features[np.array([row[1] for row in val_pos_a_p_batch])].view(-1, asp_dim, 1))
                        pos_embedding_a0 = F.elu(a_pos_embedding_list[0].view(-1, 1, asp_dim)) # [8k,1,64]

                        # p_pos_embedding = asp_features[np.array([row[1] for row in val_pos_a_p_batch])].view(-1, asp_dim, 1)

                        neg_embedding_a0 = F.elu(a_neg_embedding_list[0].view(-1, 1,asp_dim))
                        p_neg_embedding = F.elu(asp_features[np.array([row[1] for row in val_neg_a_p_batch])].view(-1, asp_dim, 1))
                        # p_neg_embedding = asp_features[np.array([row[1] for row in val_neg_a_p_batch])].view(-1, asp_dim, 1)

                        # 计算loss
                        pos_ap_out = torch.bmm(pos_embedding_a0, p_pos_embedding).view(-1, num_heads, 1).sum(dim=1)  # [64,1,1]->[8,8,1]->[8,1]
                        neg_ap_out = torch.bmm(neg_embedding_a0, p_neg_embedding).view(-1, num_heads, 1).sum(dim=1)

                        # pos_ap_max = torch.max(torch.bmm(pos_embedding_a0, p_pos_embedding).view(-1, num_heads), 1)[
                        #     1]
                        label_pos = torch.unsqueeze(torch.ones(pos_ap_out.shape[0]), -1).to(device)
                        label_neg = torch.unsqueeze(torch.zeros(pos_ap_out.shape[0]), -1).to(device)
                        ap_val_loss.append(fn_loss(pos_ap_out, label_pos) + fn_loss(neg_ap_out, label_neg))  # 一个batch的平均

                        pos_out = torch.bmm(pos_embedding_a0, p_pos_embedding).view(-1, num_heads, 1).sum(dim=1).flatten()  # 0 1维推平 [8,1]
                        neg_out = torch.bmm(neg_embedding_a0, p_neg_embedding).view(-1, num_heads, 1).sum(dim=1).flatten()

                        ap_pos_proba_list.append(torch.sigmoid(pos_out))
                        ap_neg_proba_list.append(torch.sigmoid(neg_out))

                        # 输出att

                        pos_att_list_a0_0.append(att_pos[0][0])
                        pos_att_list_a0_1.append(att_pos[0][1])
                        pos_att_list_a0_2.append(att_pos[0][2])
                        neg_att_list_a0_0.append(att_neg[0][0])
                        neg_att_list_a0_1.append(att_neg[0][1])
                        neg_att_list_a0_2.append(att_neg[0][2])

                        # 输出embedding

                        a_val_pos_embedding.append(F.elu(a_pos_embedding_list[0]))
                        a_val_neg_embedding.append(F.elu(a_neg_embedding_list[0]))
                        p_val_pos_embedding.append(F.elu(asp_features[np.array([row[1] for row in val_pos_a_p_batch])]))
                        p_val_neg_embedding.append(F.elu(asp_features[np.array([row[1] for row in val_neg_a_p_batch])]))
                        # pos_max.append(pos_ap_max)
                    ap_y_proba_test = torch.cat(ap_pos_proba_list + ap_neg_proba_list)
                    ap_y_proba_test = ap_y_proba_test.cpu().numpy()

                    ap_val_loss = torch.mean(torch.tensor(ap_val_loss))

                    pos_att_list_a0_0 = torch.cat(pos_att_list_a0_0).cpu().numpy()
                    pos_att_list_a0_1 = torch.cat(pos_att_list_a0_1).cpu().numpy()
                    pos_att_list_a0_2 = torch.cat(pos_att_list_a0_2).cpu().numpy()
                    neg_att_list_a0_0 = torch.cat(neg_att_list_a0_0).cpu().numpy()
                    neg_att_list_a0_1 = torch.cat(neg_att_list_a0_1).cpu().numpy()
                    neg_att_list_a0_2 = torch.cat(neg_att_list_a0_2).cpu().numpy()


                             # se_max=torch.cat(pos_max, 0).cpu().detach().numpy())

                t_end = time.time()

                ap_auc = roc_auc_score(y_true_test_p, ap_y_proba_test)
                ap_ap = average_precision_score(y_true_test_p, ap_y_proba_test)
                print('Link Prediction Test')

                print('ap_AUC = {}'.format(ap_auc))
                print('ap_AP = {}'.format(ap_ap))

                result_file.write('AP--' + 'auc:' + str(ap_auc) + ' ' + 'ap:' + str(ap_ap) + '\n')

                print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                    epoch, ap_val_loss.item(), t_end - t_start))


                if ap_auc > current_auc:  
                    current_auc = ap_auc
                early_stopping(ap_auc, net)
                if early_stopping.early_stop:
                    print('Early stopping!')
                    break

        # 跳出epoch
        test_idx_generator = index_generator(batch_size=batch_size, num_data=len(test_pos_a_a), shuffle=False)
        net.load_state_dict(torch.load('v1_checkpoint/checkpoint_{}.pt'.format(save_postfix)))
        net.eval()
        pos_proba_list = []
        neg_proba_list = []
        with torch.no_grad():
            for iteration in range(test_idx_generator.num_iterations()):
                # forward
                test_idx_batch = test_idx_generator.next()
                test_pos_a_a_batch = test_pos_a_a[test_idx_batch].tolist()
                test_neg_a_a_batch = test_neg_a_a[test_idx_batch].tolist()
                test_pos_g_lists, test_pos_indices_lists, test_pos_text_indices_lists, test_pos_idx_batch_mapped_lists, test_pos_nodes_lists = parse_minibatch_DBLP(
                    adjlists_aa, edge_metapath_indices_list_aa, test_pos_a_a_batch, text_mask, topic_array, device, samples=neighbor_samples, use_masks=no_masks)
                test_neg_g_lists, test_neg_indices_lists, test_neg_text_indices_lists, test_neg_idx_batch_mapped_lists, test_neg_nodes_lists = parse_minibatch_DBLP(
                    adjlists_aa, edge_metapath_indices_list_aa, test_neg_a_a_batch, text_mask, topic_array, device, samples=neighbor_samples, use_masks=no_masks)

                a_pos_embedding_list, a_neg_embedding_list, a_logits_pos_embedding_list, a_logits_neg_embedding_list, asp_features, att_pos, att_neg = net(
                    ([test_pos_g_lists,test_neg_g_lists], features_list, topic, [test_pos_text_indices_lists, test_neg_text_indices_lists], type_mask,
                     [test_pos_indices_lists, test_neg_indices_lists], [test_pos_idx_batch_mapped_lists, test_neg_idx_batch_mapped_lists], [test_pos_nodes_lists, test_neg_nodes_lists]))
                # a_neg_embedding_list, _ = net(
                #     (test_neg_g_lists, features_list, topic, test_neg_text_indices_lists, type_mask, test_neg_indices_lists, test_neg_idx_batch_mapped_lists, test_neg_nodes_lists, True))
                pos_embedding_a0 = a_pos_embedding_list[0].view(-1, 1,asp_dim)
                pos_embedding_a1 = a_pos_embedding_list[1].view(-1, asp_dim,1)

                neg_embedding_a0 = a_neg_embedding_list[0].view(-1, 1,asp_dim)
                neg_embedding_a1 = a_neg_embedding_list[1].view(-1, asp_dim,1)

                pos_out = torch.max(torch.bmm(pos_embedding_a0, pos_embedding_a1).view(-1, num_heads, 1), 1)[
                    0].flatten()  # 0 1维推平 [8,1]
                neg_out = torch.max(torch.bmm(neg_embedding_a0, neg_embedding_a1).view(-1, num_heads, 1), 1)[
                    0].flatten()

                pos_proba_list.append(torch.sigmoid(pos_out))
                neg_proba_list.append(torch.sigmoid(neg_out))
            y_proba_test = torch.cat(pos_proba_list + neg_proba_list)
            y_proba_test = y_proba_test.cpu().numpy()
        auc = roc_auc_score(y_true_test, y_proba_test)
        ap = average_precision_score(y_true_test, y_proba_test)
        print('Link Prediction Test')
        print('AUC = {}'.format(auc))
        print('AP = {}'.format(ap))
        auc_list.append(auc)
        ap_list.append(ap)

    print('----------------------------------------------------------------')
    print('Link Prediction Tests Summary')
    print('AUC_mean = {}, AUC_std = {}'.format(np.mean(auc_list), np.std(auc_list)))
    print('AP_mean = {}, AP_std = {}'.format(np.mean(ap_list), np.std(ap_list)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MRGNN testing for the DBLP dataset')
    ap.add_argument('--feats-type', type=int, default=0,  # 原本只用了author的feature 2
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2.')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--asp-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--attn-vec-dim', type=int, default=128, help='Dimension of the attention vector. Default is 128.')
    ap.add_argument('--rnn-type', default='RotatE0', help='Type of the aggregator. Default is RotatE0.')
    ap.add_argument('--epoch', type=int, default=100, help='Number of epochs. Default is 100.')
    ap.add_argument('--patience', type=int, default=5, help='Patience. Default is 5.')
    ap.add_argument('--batch-size', type=int, default=8, help='Batch size. Default is 8.')
    ap.add_argument('--neg_num', type=int, default=10, help='Number of negtives sampled. Default is 10.')
    ap.add_argument('--samples', type=int, default=100, help='Number of neighbors sampled. Default is 100.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--save-postfix', default='DBLP', help='Postfix for the saved model and result. Default is DBLP.')
    ap.add_argument('--gpu_num', type=int, default=0)
    ap.add_argument('--random_seed', type=int, default=1)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--lamda', type=float, default=1e-3)
    ap.add_argument('--ga', type=float, default=1e-3)
    ap.add_argument('--max_iter', type=int, default=3)
    ap.add_argument('--pr', type=int, default=0)
    ap.add_argument('--prc', type=str, default='mean')
    args = ap.parse_args()
    # fix random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    run_model_DBLP(args.feats_type, args.hidden_dim, args.asp_dim, args.num_heads, args.attn_vec_dim, args.rnn_type,
                   args.epoch, args.patience, args.batch_size, args.samples, args.neg_num, args.repeat, args.save_postfix,args.gpu_num,args.random_seed,args.lr, args.max_iter,args.dropout, args.lamda, args.ga,args.pr,args.prc)
