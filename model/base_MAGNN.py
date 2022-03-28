import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax
import copy


class MAGNN_metapath_instance(nn.Module):
    '''calculate instances'''
    def __init__(self,
                 etypes,  # 每条metapath上的边类型
                 out_dim,  # 64 asp_dim
                 num_heads,
                 rnn_type='gru',
                 r_vec=None,  # ?
                 max_iter=3,
                 attn_drop=0.5,
                 alpha=0.01,
                 use_minibatch=False,
                 attn_switch=False):
        super(MAGNN_metapath_instance, self).__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.d = self.num_heads * self.out_dim
        self.rnn_type = rnn_type
        self.etypes = etypes
        self.r_vec = r_vec
        self.use_minibatch = use_minibatch
        self.attn_switch = attn_switch  # 权重初始化
        self.max_iter = max_iter

        if rnn_type == 'gru':
            self.rnn = nn.GRU(out_dim, num_heads * out_dim)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(out_dim, num_heads * out_dim)
        elif rnn_type == 'bi-gru':
            self.rnn = nn.GRU(out_dim, num_heads * out_dim // 2, bidirectional=True)
        elif rnn_type == 'bi-lstm':
            self.rnn = nn.LSTM(out_dim, num_heads * out_dim // 2, bidirectional=True)
        elif rnn_type == 'linear':
            self.rnn = nn.Linear(out_dim, num_heads * out_dim)
        elif rnn_type == 'max-pooling':
            self.rnn = nn.Linear(out_dim, num_heads * out_dim)
        elif rnn_type == 'neighbor-linear':
            self.rnn = nn.Linear(out_dim, num_heads * out_dim)

        self.softmax = edge_softmax
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x




    def forward(self, inputs):
        # features: num_all_nodes x out_dim
        if self.use_minibatch:  # 如果有batch，有idx新图
            g, features, topic, edge_metapath_text_indices, type_mask, edge_metapath_indices, target_idx, node = inputs
        else:
            g, features, topic, edge_metapath_text_indices, type_mask, edge_metapath_indices = inputs

        # Embedding layer
        # use torch.nn.functional.embedding or torch.embedding here
        # do not use torch.nn.embedding
        # edata: E x Seq x out_dim
        features = features.view(-1, self.d)
        edata = F.embedding(edge_metapath_indices, features)

        # apply rnn to metapath-based feature sequence
        if self.rnn_type == 'gru':
            _, hidden = self.rnn(edata.permute(1, 0, 2))  # 交换维度 改的话需要第一维全堆在一起，过一个rnn再view
        elif self.rnn_type == 'lstm':
            _, (hidden, _) = self.rnn(edata.permute(1, 0, 2))
        elif self.rnn_type == 'bi-gru':
            _, hidden = self.rnn(edata.permute(1, 0, 2))
            hidden = hidden.permute(1, 0, 2).reshape(-1, self.out_dim, self.num_heads).permute(0, 2, 1).reshape(
                -1, self.num_heads * self.out_dim).unsqueeze(dim=0)
        elif self.rnn_type == 'bi-lstm':
            _, (hidden, _) = self.rnn(edata.permute(1, 0, 2))
            hidden = hidden.permute(1, 0, 2).reshape(-1, self.out_dim, self.num_heads).permute(0, 2, 1).reshape(
                -1, self.num_heads * self.out_dim).unsqueeze(dim=0)
        elif self.rnn_type == 'average':  # 已改
            hidden = torch.mean(edata[:, :-1, :], dim=1)
            # hidden = torch.mean(edata, dim=1)
            hidden = hidden.view(-1, self.num_heads, self.out_dim)  # [782,8,64]
            # hidden = torch.cat([hidden] * self.num_heads, dim=1)   # 为何在这里直接拼接多头  [782,8,64]
            hidden = hidden.unsqueeze(dim=0)  # [1,782,8,64]
        elif self.rnn_type == 'linear':
            hidden = self.rnn(torch.mean(edata, dim=1))
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'max-pooling':
            hidden, _ = torch.max(self.rnn(edata), dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'TransE0' or self.rnn_type == 'TransE1':
            r_vec = self.r_vec
            if self.rnn_type == 'TransE0':
                r_vec = torch.stack((r_vec, -r_vec), dim=1)
                r_vec = r_vec.reshape(self.r_vec.shape[0] * 2, self.r_vec.shape[1])  # etypes x out_dim
            edata = F.normalize(edata, p=2, dim=2)
            for i in range(edata.shape[1] - 1):
                # consider None edge (symmetric relation)
                temp_etypes = [etype for etype in self.etypes[i:] if etype is not None]
                edata[:, i] = edata[:, i] + r_vec[temp_etypes].sum(dim=0)
            hidden = torch.mean(edata, dim=1)
            hidden = torch.cat([hidden] * self.num_heads, dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'RotatE0' or self.rnn_type == 'RotatE1':
            r_vec = F.normalize(self.r_vec, p=2, dim=2)  # 先正则
            if self.rnn_type == 'RotatE0':
                r_vec = torch.stack((r_vec, r_vec), dim=1)
                r_vec[:, 1, :, 1] = -r_vec[:, 1, :, 1]
                r_vec = r_vec.reshape(self.r_vec.shape[0] * 2, self.r_vec.shape[1], 2)  # etypes x out_dim/2 x 2
            edata = edata.reshape(edata.shape[0], edata.shape[1], edata.shape[2] // 2, 2)  # embedding割成两半
            final_r_vec = torch.zeros([edata.shape[1], self.out_dim // 2, 2], device=edata.device)
            final_r_vec[-1, :, 0] = 1
            for i in range(final_r_vec.shape[0] - 2, -1, -1):
                # consider None edge (symmetric relation)
                if self.etypes[i] is not None:
                    final_r_vec[i, :, 0] = final_r_vec[i + 1, :, 0].clone() * r_vec[self.etypes[i], :, 0] - \
                                           final_r_vec[i + 1, :, 1].clone() * r_vec[self.etypes[i], :, 1]
                    final_r_vec[i, :, 1] = final_r_vec[i + 1, :, 0].clone() * r_vec[self.etypes[i], :, 1] + \
                                           final_r_vec[i + 1, :, 1].clone() * r_vec[self.etypes[i], :, 0]
                else:
                    final_r_vec[i, :, 0] = final_r_vec[i + 1, :, 0].clone()
                    final_r_vec[i, :, 1] = final_r_vec[i + 1, :, 1].clone()
            for i in range(edata.shape[1] - 1):
                temp1 = edata[:, i, :, 0].clone() * final_r_vec[i, :, 0] - \
                        edata[:, i, :, 1].clone() * final_r_vec[i, :, 1]
                temp2 = edata[:, i, :, 0].clone() * final_r_vec[i, :, 1] + \
                        edata[:, i, :, 1].clone() * final_r_vec[i, :, 0]
                edata[:, i, :, 0] = temp1
                edata[:, i, :, 1] = temp2
            edata = edata.reshape(edata.shape[0], edata.shape[1], -1)
            hidden = torch.mean(edata, dim=1)
            hidden = torch.cat([hidden] * self.num_heads, dim=1)  # 直接复制
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'neighbor':
            hidden = edata[:, 0]  # [782,64] 第一个是邻居?  找找什么时候换的顺序 有target索引但存的时候是反向存储 因为尾节点是src,头节点才是dst
            hidden = torch.cat([hidden] * self.num_heads, dim=1)
            hidden = hidden.unsqueeze(dim=0)
        elif self.rnn_type == 'neighbor-linear':
            hidden = self.rnn(edata[:, 0])
            hidden = hidden.unsqueeze(dim=0)

        eft = F.normalize(hidden.permute(1, 0, 2, 3).view(-1, self.num_heads, self.out_dim))  # E x num_heads x out_dim

        return eft





class MAGNN_metapath_specific(nn.Module):
    '''aggregate in metapath'''
    def __init__(self,
                 etypes,  # 每条metapath上的边类型
                 out_dim,  # 64 asp_dim
                 num_heads,
                 use_minibatch=False,
                 attn_switch=False):
        super(MAGNN_metapath_specific, self).__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.d = self.num_heads * self.out_dim
        self.etypes = etypes
        self.use_minibatch = use_minibatch
        self.attn_switch = attn_switch  # 权重初始化


        self.softmax = edge_softmax






    def edge_softmax(self, g):   # 这里softmax是edge softmax 我们直接在边上做softmax
        attention = self.softmax(g, g.edata.pop('a'))  # 返回边字典的a的value,做softmax, 并且从字典中删除
        # print('attention',attention.shape) [782,8,1] 782条边，根据目标节点做softmax,8套参数 输入是[E,*,1],输出相同
        # 这里要改，在边上做softmax
        # edge_softmax会认边的方向，一边根据目标节点去做softmax，所以在构图时要特别注意，左边是源节点，右边才是目标节点
        # Dropout attention scores and save them
        g.edata['a_drop'] = self.attn_drop(attention)  # 计算了概率之后还dropout

    def message_passing(self, edges):

        ft = edges.data['eft'] * edges.data['a']  # 边的两种属性相乘得到节点特征,edges.data=edata,只是为了区分同质图和异质图
        return {'nft': ft}

    def forward(self, inputs):
        # features: num_all_nodes x out_dim
        if self.use_minibatch:  # 如果有batch，有idx新图
            g, features, topic, edge_metapath_text_indices, type_mask, eft, target_idx, node, start = inputs
        else:
            g, features, topic, edge_metapath_text_indices, type_mask, eft, start = inputs

        # Embedding layer
        # use torch.nn.functional.embedding or torch.embedding here
        # do not use torch.nn.embedding
        # edata: E x Seq x out_dim

        # node = F.normalize(F.embedding(node, features).view(-1, self.num_heads, self.out_dim), 2)
        g.ndata.update({'ft': node})  # target节点个数
        g.edata.update({'eft': eft})  # 实例

        g.apply_edges(fn.e_mul_v('eft', 'ft', 'm'))  # v不等于ft  复制到边上 [240,16,16]
        # _cache_zero_k = torch.zeros(1, self.num_heads).to(node.device)
        # if start==True:  # 热启动
        #     a = F.softmax(_cache_zero_k.expand(eft.shape[0], self.num_heads), dim=-1)  # [240,16]
        #     a = torch.div(a, torch.sqrt(a.sum(dim=0, keepdim=True)+1e-9)).unsqueeze(dim=-1)
        # else:
        sim = g.edata.pop('m').sum(dim=-1)   # [240,16]
        # 有可能是空的
        #max_sim=torch.max(sim, 1)[0].unsqueeze(dim=-1)

        a = F.softmax(sim, dim=-1).unsqueeze(dim=-1)  # [240,16,1]
        # print(a)
        # a = torch.div(a, torch.sqrt(a.sum(dim=0, keepdim=True) + 1e-9))
        g.edata.update({'a': a})

        if g.number_of_edges() == 0:
            g.ndata['nft'] = torch.zeros(
                node.shape[0], node.shape[1], node.shape[2]).to(node.device)  # 没有边就跟node一样
        else:

            g.update_all(self.message_passing, fn.sum('nft', 'nft'))  # 触发消息传递，ft+message赋给ft,更新特征
        v = g.ndata['nft'].view(-1, self.num_heads, self.out_dim)


        if self.use_minibatch:
            return [F.normalize(v[target_idx]+1e-15,2), a]  # [8,16,16]  找到batch里的target节点 因为只有target节点是被更新的。
        else:
            return v
# 没有leaky_relu比较稳定

class MAGNN_ctr_ntype_specific(nn.Module):  # 对一种类型节点，不同元路径的聚合
    def __init__(self,
                 num_metapaths,
                 etypes_list,
                 out_dim,   # asp_dim
                 num_heads,
                 attn_vec_dim,
                 max_iter,
                 rnn_type='gru',
                 r_vec=None,
                 attn_drop=0.5,
                 use_minibatch=False):
        super(MAGNN_ctr_ntype_specific, self).__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.use_minibatch = use_minibatch
        self.max_iter = max_iter


        # metapath-specific layers
        self.metapath_instances = nn.ModuleList()
        self.metapath_layers = nn.ModuleList()  # 在init过程中用list

        for i in range(num_metapaths):
            self.metapath_instances.append(
                MAGNN_metapath_instance(etypes_list[i],  # 3条不同元路径使用不同的边集合  metapath_specific模型只有边集合不同
                                        out_dim,  # asp_dim
                                        num_heads,
                                        rnn_type,
                                        r_vec,
                                        max_iter,
                                        attn_drop=attn_drop,
                                        use_minibatch=use_minibatch))
            self.metapath_layers.append(MAGNN_metapath_specific(etypes_list[i],  # 3条不同元路径使用不同的边集合  metapath_specific模型只有边集合不同
                                                                out_dim,  # asp_dim
                                                                num_heads,
                                                                use_minibatch=use_minibatch))

        # metapath-level attention
        # note that the acutal input dimension should consider the number of heads
        # as multiple head outputs are concatenated together
        # 元路径level的attention
        self.fc1 = nn.Linear(out_dim, out_dim, bias=False)  # MA bA   128  结构的选择   k个w
        self.fc2 = nn.Parameter(torch.empty(size=(num_heads, out_dim)))  # qA

        # weight initialization
        nn.init.xavier_uniform_(self.fc1.weight, gain=1.414)
        nn.init.xavier_uniform_(self.fc2.data, gain=1.414)

    def forward(self, inputs):
        if self.use_minibatch:  # 如果使用batch，那就要再输入一个map了目标节点与邻居节点的list,构造小图
            g_list, features, topic, edge_metapath_text_indices_list, type_mask, edge_metapath_indices_list, target_idx_list, node_list = inputs
            instance_outs = [metapath_instance(
                (g, features, topic, edge_metapath_text_indices, type_mask, edge_metapath_indices, target_idx, node))
                             for g, edge_metapath_text_indices, edge_metapath_indices, target_idx, node, metapath_instance
                             in zip(g_list, edge_metapath_text_indices_list, edge_metapath_indices_list, target_idx_list,
                                 node_list, self.metapath_instances)]  # 实例 normalization
            u = None
            metapath_att_outs=None
            # node_list代表三个图中所有节点原来的索引

            for i in range(self.max_iter):  # 直接用目标节点的特征
                if i == 0:
                    node_embedding_list = [
                        F.normalize(F.embedding(node, features).view(-1, self.num_heads, self.out_dim), 2)
                        for node in node_list]  # 三个图的所有节点embedding
                    metapath_outs = [metapath_layer((g, features, topic, edge_metapath_text_indices, type_mask,
                                                     edge_metapath_indices, target_idx, node, True))
                                     for
                                     g, edge_metapath_text_indices, edge_metapath_indices, target_idx, node, metapath_layer
                                     in zip(g_list, edge_metapath_text_indices_list, instance_outs, target_idx_list,
                                            node_embedding_list, self.metapath_layers)]

                else:
                    node_feature_list = [torch.zeros_like(embedding) for embedding in node_embedding_list]
                    for path in range(len(g_list)):  # 分别送到三个图中
                        node_feature_list[path][target_idx_list[path]] = u  # 修改矩阵元素 整个替换就不会无法求梯度
                    node_embedding_list = node_feature_list
                    metapath_outs = [metapath_layer((g, features, topic, edge_metapath_text_indices, type_mask,
                                                     edge_metapath_indices, target_idx, node, False))
                                     for
                                     g, edge_metapath_text_indices, edge_metapath_indices, target_idx, node, metapath_layer
                                     in zip(g_list, edge_metapath_text_indices_list, instance_outs, target_idx_list,
                                            node_embedding_list, self.metapath_layers)]

                metapath_embedding_outs = [metapath_out[0].view(-1, self.num_heads * self.out_dim) for metapath_out in metapath_outs]
                metapath_att_outs = [metapath_out[1] for metapath_out in metapath_outs]

                beta = []
                factor = torch.FloatTensor([0.0]).to(metapath_att_outs[0].device)

                for metapath_out in metapath_embedding_outs:
                #
                #     fc1 = torch.tanh(self.fc1(F.elu(metapath_out).view(-1, self.num_heads, self.out_dim)))  # 8*128
                #     fc1_mean = torch.mean(fc1, dim=0)  # [16,16]
                #
                #     fc2 = torch.unsqueeze((self.fc2 * fc1_mean).sum(dim=1).sum(dim=0),dim=-1)   # epi [K,1]
                    beta.append(factor)

                beta = torch.cat(beta, dim=0)
                # print(beta)
                # max_beta = torch.max(beta, 1)[0].unsqueeze(dim=-1)
                beta = F.softmax(beta, dim=0)
                # print(beta)


                beta = torch.unsqueeze(beta, dim=-1)
                beta = torch.unsqueeze(beta, dim=-1)
                # print("beta",beta.shape) [3,1,1]
                metapath_outs = [torch.unsqueeze(metapath_out, dim=0) for metapath_out in
                                 metapath_embedding_outs]
                metapath_outs = torch.cat(metapath_outs, dim=0)  # [3, 8, 256]

                h = torch.sum(beta * metapath_outs, dim=0).view(-1, self.num_heads, self.out_dim)  # [8,16,16]
                # 更新target node embedding 只更新一个图
                # if i < self.max_iter-1:
                u = F.normalize(node_embedding_list[0][target_idx_list[0]] + h)
                # else:
                #     u=node_embedding_list[0][target_idx_list[0]] + h



        else:
                g_list, features, topic, edge_metapath_text_indices_list, type_mask, edge_metapath_indices_list = inputs

                # metapath-specific layers
                metapath_outs = [metapath_layer((g, features, topic, edge_metapath_text_indices, type_mask, edge_metapath_indices)).view(-1, self.num_heads * self.out_dim)
                                 for g, edge_metapath_text_indices, edge_metapath_indices, metapath_layer in zip(g_list, edge_metapath_text_indices_list, edge_metapath_indices_list, self.metapath_layers)]


        return u, metapath_att_outs#, torch.squeeze(beta, dim=-1)