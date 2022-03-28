import torch
import dgl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
import random


def idx_to_one_hot(idx_arr):
    one_hot = np.zeros((idx_arr.shape[0], idx_arr.max() + 1))
    one_hot[np.arange(idx_arr.shape[0]), idx_arr] = 1
    return one_hot


def kmeans_test(X, y, n_clusters, repeat=10):
    nmi_list = []
    ari_list = []
    for _ in range(repeat):
        kmeans = KMeans(n_clusters=n_clusters)
        y_pred = kmeans.fit_predict(X)
        nmi_score = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
        ari_score = adjusted_rand_score(y, y_pred)
        nmi_list.append(nmi_score)
        ari_list.append(ari_score)
    return np.mean(nmi_list), np.std(nmi_list), np.mean(ari_list), np.std(ari_list)


def svm_test(X, y, test_sizes=(0.2, 0.4, 0.6, 0.8), repeat=10):
    random_states = [182318 + i for i in range(repeat)]
    result_macro_f1_list = []
    result_micro_f1_list = []
    for test_size in test_sizes:
        macro_f1_list = []
        micro_f1_list = []
        for i in range(repeat):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=True, random_state=random_states[i])
            svm = LinearSVC(dual=False)
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            macro_f1 = f1_score(y_test, y_pred, average='macro')
            micro_f1 = f1_score(y_test, y_pred, average='micro')
            macro_f1_list.append(macro_f1)
            micro_f1_list.append(micro_f1)
        result_macro_f1_list.append((np.mean(macro_f1_list), np.std(macro_f1_list)))
        result_micro_f1_list.append((np.mean(micro_f1_list), np.std(micro_f1_list)))
    return result_macro_f1_list, result_micro_f1_list


def evaluate_results_nc(embeddings, labels, num_classes):
    print('SVM test')
    svm_macro_f1_list, svm_micro_f1_list = svm_test(embeddings, labels)
    print('Macro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(macro_f1_mean, macro_f1_std, train_size) for
                                    (macro_f1_mean, macro_f1_std), train_size in
                                    zip(svm_macro_f1_list, [0.8, 0.6, 0.4, 0.2])]))
    print('Micro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(micro_f1_mean, micro_f1_std, train_size) for
                                    (micro_f1_mean, micro_f1_std), train_size in
                                    zip(svm_micro_f1_list, [0.8, 0.6, 0.4, 0.2])]))
    print('K-means test')
    nmi_mean, nmi_std, ari_mean, ari_std = kmeans_test(embeddings, labels, num_classes)
    print('NMI: {:.6f}~{:.6f}'.format(nmi_mean, nmi_std))
    print('ARI: {:.6f}~{:.6f}'.format(ari_mean, ari_std))

    return svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std


def parse_adjlist(adjlist, edge_metapath_indices, samples=None):
    edges = []
    nodes = set()
    result_indices = []
    for row, indices in zip(adjlist, edge_metapath_indices):
        row_parsed = list(map(int, row.split(' ')))
        nodes.add(row_parsed[0])
        if len(row_parsed) > 1:
            # sampling neighbors
            if samples is None:
                neighbors = row_parsed[1:]
                result_indices.append(indices)
            else:
                # undersampling frequent neighbors
                unique, counts = np.unique(row_parsed[1:], return_counts=True)
                p = []
                for count in counts:
                    p += [(count ** (3 / 4)) / count] * count
                p = np.array(p)
                p = p / p.sum()
                samples = min(samples, len(row_parsed) - 1)
                sampled_idx = np.sort(np.random.choice(len(row_parsed) - 1, samples, replace=False, p=p))
                neighbors = [row_parsed[i + 1] for i in sampled_idx]
                result_indices.append(indices[sampled_idx])
        else:
            neighbors = []  # 没有邻居就不连了
            result_indices.append(indices)
        for dst in neighbors:
            nodes.add(dst)
            edges.append((row_parsed[0], dst))
    mapping = {map_from: map_to for map_to, map_from in enumerate(sorted(nodes))}
    edges = list(map(lambda tup: (mapping[tup[0]], mapping[tup[1]]), edges))
    result_indices = np.vstack(result_indices)
    return edges, result_indices, len(nodes), mapping


def parse_minibatch(adjlists, edge_metapath_indices_list, idx_batch, device, samples=None):
    g_list = []
    result_indices_list = []
    idx_batch_mapped_list = []
    for adjlist, indices in zip(adjlists, edge_metapath_indices_list):
        edges, result_indices, num_nodes, mapping = parse_adjlist(
            [adjlist[i] for i in idx_batch], [indices[i] for i in idx_batch], samples)

        g = dgl.DGLGraph(multigraph=True)
        g.add_nodes(num_nodes)
        if len(edges) > 0:
            sorted_index = sorted(range(len(edges)), key=lambda i : edges[i])
            g.add_edges(*list(zip(*[(edges[i][1], edges[i][0]) for i in sorted_index])))
            result_indices = torch.LongTensor(result_indices[sorted_index]).to(device)
        else:
            result_indices = torch.LongTensor(result_indices).to(device)
        #g.add_edges(*list(zip(*[(dst, src) for src, dst in sorted(edges)])))
        #result_indices = torch.LongTensor(result_indices).to(device)
        g_list.append(g)
        result_indices_list.append(result_indices)
        idx_batch_mapped_list.append(np.array([mapping[idx] for idx in idx_batch]))

    return g_list, result_indices_list, idx_batch_mapped_list

def parse_adjlist_DBLP(adjlist, edge_metapath_indices, text_mask, topic_array, indices_len, off,samples=None, exclude=None):
    edges = []
    nodes = set()  # 集合
    result_indices = []
    result_indices_neg = []
    text_indices = []

    # off = 11569  # 13197
    topic_samples = 30
    for row, indices in zip(adjlist, edge_metapath_indices):  # 遍历batch中的样本
        row_parsed = list(map(int, row.split(' ')))  # 所有邻居 
        text_indice = []
        nodes.add(row_parsed[0])

        if len(indices) > 0:  # 若有邻居样例
            # sampling neighbors
            sampled_idx = np.arange(0, len(indices))

            sample_indices = indices
            print(len(text_mask))
            if len(text_mask) > 1:  # 长的元路径
                topic_samples = min(topic_samples, len(indices))   # 只剩30个邻居
                # cos
                alpha = topic_array[indices[:, text_mask[0]] - off]
                beta = topic_array[indices[:, text_mask[1]] - off]


                p = (alpha * beta).sum(1) / (np.linalg.norm(alpha, axis=1) * np.linalg.norm(beta, axis=1)+1e-9)

                p = np.clip(p, 0, 0.9)  
                p = p+1e-9
                p = p / p.sum() 

                sampled_idx = np.sort(np.random.choice(len(indices), topic_samples, replace=False, p=p))
                sample_indices = indices[sampled_idx]


            if exclude is not None:  # mask 说明是训练集
                mask = [False if [a1, p1] in exclude or [a2, p2] in exclude else True for
                            a1, p1, a2, p2 in sample_indices[:, [0, 1, -1, -2]]]
                neighbors = np.array([indices[:, 0][i] for i in sampled_idx])[mask]
                result_indices.append(sample_indices[mask])  # 更有可能是mask掉之后没有邻居

                if len(sample_indices[mask]) > 0:

                    text_indice = np.expand_dims(sample_indices[mask][:, text_mask[-1]] - off, axis=1)
                else:
                    text_indice = np.expand_dims(np.array([]), axis=1)


            else:  #
                neighbors = [indices[:, 0][i] for i in sampled_idx]  # 所采的邻居 同类型
                result_indices.append(sample_indices)  # append每一个样本所采邻居对应的元路径实例 [[×××],[×××],[×××]]len<=100
                text_indice = np.expand_dims(sample_indices[:, text_mask[-1]] - off, axis=1)

             # 每个样本所有样例，两列 取第一列就行
            text_indices.append(text_indice)

        else:
            neighbors = [row_parsed[0]]
            # np.array([[row_parsed[0]] * indices.shape[1]])
            result_indices.append([[row_parsed[0]] * indices_len])   # 没有邻居就不连
            text_indices.append(np.array([-1]))  
            # if len(text_mask) > 1:
            #     text_indices.append(indices.reshape(0, 2))     # 可以加入头节点作为邻居 但是text_indices就不要加
            # else:
            #     text_indices.append(indices.reshape(0, 1))
        for dst in neighbors:        # 有可能为空
            nodes.add(dst)  # nodes是集合 不会重复 包含基于该元路径的所有尾节点邻居
            edges.append((row_parsed[0], dst))  # 只有目标节点和邻居节点的连边 有可能整个图没边

    nodes = sorted(nodes)   # 记住原来序号
    mapping = {map_from: map_to for map_to, map_from in enumerate(nodes)}  # 节点一一映射成从0开始的index
    edges = list(map(lambda tup: (mapping[tup[0]], mapping[tup[1]]), edges))  # 同样映射一下 tup是edges里的元素，两个节点分别映射
    result_indices = np.vstack(result_indices)  # 不管一个样本采了多少邻居，全都堆叠一起，当成边的属性
    text_indices = np.vstack(text_indices)



    return edges, result_indices, text_indices, len(nodes), mapping, np.expand_dims(np.array(nodes),1)

def parse_minibatch_DBLP(adjlists_aa, edge_metapath_indices_list_aa, a_a_batch, text_masks, topic_array, device, off =11596,a_p_batch=None, samples=None, use_masks=None, modes=2):
    g_lists = []  # aa
    result_indices_lists = []
    text_indices_lists = []
    idx_batch_mapped_lists = []
    nodes_lists = []
    # 只有一种类型不用mode 去掉第一层循环
    # 不用mask

    for mode in range(modes):  # 这里mode指左右
        g_lists.append([])
        result_indices_lists.append([])
        text_indices_lists.append([])
        idx_batch_mapped_lists.append([])
        nodes_lists.append([])
        for adjlist, indices, use_mask, text_mask, indices_len in zip(adjlists_aa, edge_metapath_indices_list_aa, use_masks, text_masks, [3, 5, 5]):  # u的每一条元路径对应的邻居和实例

            if use_mask:
                edges, result_indices, text_indices, num_nodes, mapping, nodes = parse_adjlist_DBLP(
                    [adjlist[row[mode]] for row in a_a_batch],  # adjlist有问题
                    [indices[row[mode]] for row in a_a_batch], text_mask, topic_array, indices_len, off, samples, a_p_batch)
            else:
                edges, result_indices, text_indices, num_nodes, mapping, nodes = parse_adjlist_DBLP(
                    [adjlist[row[mode]] for row in a_a_batch],
                    [indices[row[mode]] for row in a_a_batch], text_mask, topic_array, indices_len, off, samples)
        # 允许重边
            g = dgl.DGLGraph(multigraph=True)  # 必须是多重图 一个样本会有多个相同邻居
            g.add_nodes(num_nodes)   # 节点个数，从0开始
            if len(edges) > 0:  # 如果有边  这里暂时理解有误，应该是一个batch构的一个图
                sorted_index = sorted(range(len(edges)), key=lambda i : edges[i])   # 按照边原来的顺序对range(len(edges))排序 节点小的排在前面
                g.add_edges(*list(zip(*[(edges[i][1], edges[i][0]) for i in sorted_index])))  # *号是去掉list的括号 变成从源节点指向target节点
                result_indices = torch.LongTensor(result_indices[sorted_index]).to(device)
                text_indices = torch.LongTensor(text_indices[sorted_index]).to(device)   # 可能遇到空
            else:
                result_indices = torch.LongTensor(result_indices).to(device)
                text_indices = torch.LongTensor(text_indices).to(device)
            nodes = torch.LongTensor(nodes).to(device)
            g_lists[mode].append(g.to(device))  # 图没有包装成torch, g_lists指的是一个batch  [2,3]  一个batch中两种节点，各三条元路径
            result_indices_lists[mode].append(result_indices)  # mode表示左a右a [2 3] 元路径实例 一个batch的所有元路径实例都堆在一起
            text_indices_lists[mode].append(text_indices)
            idx_batch_mapped_lists[mode].append(np.array([mapping[row[mode]] for row in a_a_batch]))  # 映射成同质图中的index
            nodes_lists[mode].append(nodes)
    return g_lists, result_indices_lists, text_indices_lists, idx_batch_mapped_lists, nodes_lists



def get_ap_negtive_nodes(train_nodes, adj_lists, nodes, num_neg):  # 直接采没写过的,和未来不重叠
    node_negtive_pairs = []
    for n, node in enumerate(nodes):
        neg_nodes = set(train_nodes)-set(adj_lists[int(node)])
        neg_samples = random.sample(neg_nodes, num_neg)
        node_negtive_pairs.append(neg_samples)
        # node_negtive_pairs[node] = [(node, neg_node) for neg_node in neg_samples]
    return node_negtive_pairs


# ✔
class index_generator:  # 样本总数量，正负样本分开，正样本需要打乱
    def __init__(self, batch_size, num_data=None, indices=None, shuffle=True):
        if num_data is not None:
            self.num_data = num_data
            self.indices = np.arange(num_data)
        if indices is not None:  # 如果没有输入indice，就由num_data决定
            self.num_data = len(indices)
            self.indices = np.copy(indices)
        self.batch_size = batch_size
        self.iter_counter = 0
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.indices)

    def next(self):
        if self.num_iterations_left() <= 0:
            self.reset()
        self.iter_counter += 1  # 调用一次加1
        return np.copy(self.indices[(self.iter_counter - 1) * self.batch_size:self.iter_counter * self.batch_size]) # 深拷贝，不改变原来的值

    def num_iterations(self):
        return int(np.ceil(self.num_data / self.batch_size))  # 向上取整

    def num_iterations_left(self):
        return self.num_iterations() - self.iter_counter

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.iter_counter = 0
