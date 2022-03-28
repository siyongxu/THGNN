import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as fn


from model.base_MAGNN import MAGNN_ctr_ntype_specific


# for link prediction task
class MAGNN_lp_layer(nn.Module):
    def __init__(self,
                 num_metapaths,
                 num_edge_type,
                 etypes_list,
                 in_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 max_iter,
                 rnn_type='gru',
                 attn_drop=0.5):
        super(MAGNN_lp_layer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        # etype-specific parameters
        r_vec = None  # 边参数，为什么要empty  共享参数
        if rnn_type == 'TransE0':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type // 2, in_dim)))
        elif rnn_type == 'TransE1':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type, in_dim)))
        elif rnn_type == 'RotatE0':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type // 2, in_dim // 2, 2)))  # in_dim除以2
        elif rnn_type == 'RotatE1':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type, in_dim // 2, 2)))
        if r_vec is not None:
            nn.init.xavier_normal_(r_vec.data, gain=1.414)

        # ctr_ntype-specific layers
        self.user_layer = MAGNN_ctr_ntype_specific(num_metapaths,
                                                   etypes_list,
                                                   in_dim,  # asp_dim
                                                   num_heads,
                                                   attn_vec_dim,
                                                   max_iter,
                                                   rnn_type,
                                                   r_vec,
                                                   attn_drop,
                                                   use_minibatch=True)



        # note that the acutal input dimension should consider the number of heads
        # as multiple head outputs are concatenated together

        self.fc_user = nn.Linear(in_dim * num_heads, out_dim, bias=True)
        # self.fc_item = nn.Linear(in_dim * num_heads, out_dim, bias=True)
        nn.init.xavier_normal_(self.fc_user.weight, gain=1.414)




    def forward(self, inputs):
        g_lists, features, topic, edge_metapath_text_indices_lists, type_mask, edge_metapath_indices_lists, target_idx_lists, node_lists = inputs
        h_user_list = []
        h_logits_user_list = []
        att_list = []
        # ctr_ntype-specific layers
        for i in range(len(g_lists)):
            h_user, att = self.user_layer(
                (g_lists[i], features, topic, edge_metapath_text_indices_lists[i], type_mask, edge_metapath_indices_lists[i], target_idx_lists[i], node_lists[i]))
            # switch控制target节点
            # if switch and i == 0:
            #     h_user = torch.einsum('bcd,cde->bce', [h_user.view(-1, self.num_heads, self.in_dim), self.weight])
            #     h_user = h_user.reshape(-1, self.num_heads * self.in_dim)
            logits_user = self.fc_user(h_user.view(-1, self.in_dim * self.num_heads))
            h_user_list.append(h_user)
            h_logits_user_list.append(logits_user)
            att_list.append(att)

        return h_user_list, h_logits_user_list, att_list

class DHGNN_lp(nn.Module):
    def __init__(self,
                 num_metapaths,
                 num_edge_type,
                 etypes_list,
                 feats_dim_list,
                 asp_dim,  # 方面映射
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 max_iter,
                 rnn_type='gru',
                 dropout_rate=0.5):
        super(DHGNN_lp, self).__init__()

        # ntype-specific transformation

        self.num_heads = num_heads
        self.asp_dim = asp_dim
        self.dim = self.num_heads * self.asp_dim

        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, self.dim, bias=True) for feats_dim in feats_dim_list])
        self.cores = nn.Linear(asp_dim, num_heads, bias=False)
        # initialization of fc layers
        self.reset_parameters()



        # feature dropout after transformation
        if dropout_rate > 0:
            self.feat_drop = nn.Dropout(dropout_rate)
        else:
            self.feat_drop = lambda x: x




        # MAGNN_lp layers
        self.layer1 = MAGNN_lp_layer(num_metapaths,  # 3
                                     num_edge_type,   # 6
                                     etypes_list,  # [2,3,...]
                                     self.asp_dim,
                                     out_dim,
                                     num_heads,
                                     attn_vec_dim,
                                     max_iter,
                                     rnn_type,
                                     attn_drop=dropout_rate)
    def reset_parameters(self):
        for fc in self.fc_list:
            stdv = 1. / np.sqrt(fc.weight.size(1))
            fc.weight.data.uniform_(-stdv, stdv)
            fc.bias.data.uniform_(-stdv, stdv)

        nn.init.xavier_uniform_(self.cores.weight, gain=1.414)


    def forward(self, inputs):
        g_lists, features_list, topic, edge_metapath_text_indices_lists, type_mask, edge_metapath_indices_lists, target_idx_lists, nodes_lists = inputs

        transformed_features = torch.zeros(type_mask.shape[0], self.dim,
                                           device=features_list[0].device)

        for i, fc in enumerate(self.fc_list):
            node_indices = np.where(type_mask == i)[0] 
            transformed_features[node_indices] = fc(features_list[i])  # 不同类型的节点特征


        asp_features = self.feat_drop(transformed_features.view(-1, self.asp_dim))
        asp_features = asp_features.view(-1, self.num_heads*self.asp_dim)



        h_user_list, h_logits_user_list, att = self.layer1(
               (g_lists[0], asp_features, topic, edge_metapath_text_indices_lists[0], type_mask, edge_metapath_indices_lists[0], target_idx_lists[0], nodes_lists[0]))
        align_score = []
        for h in h_user_list:  # 没有经过elu
            align_score.append(self.cores(torch.mean(h, 0)))

        return h_user_list, h_logits_user_list, asp_features, align_score, att
