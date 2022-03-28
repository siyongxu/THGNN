import numpy as np
import scipy.sparse
import networkx as nx


def get_metapath_adjacency_matrix(adjM, type_mask, metapath):
    """
    :param M: the raw adjacency matrix
    :param type_mask: an array of types of all node
    :param metapath
    :return: a list of metapath-based adjacency matrices
    """
    out_adjM = scipy.sparse.csr_matrix(adjM[np.ix_(type_mask == metapath[0], type_mask == metapath[1])])
    for i in range(1, len(metapath) - 1):
        out_adjM = out_adjM.dot(scipy.sparse.csr_matrix(adjM[np.ix_(type_mask == metapath[i], type_mask == metapath[i + 1])]))
    return out_adjM.toarray()


# networkx.has_path may search too
def get_metapath_neighbor_pairs(M, type_mask, expected_metapaths):
    """
    :param M: the raw adjacency matrix
    :param type_mask: an array of types of all node
    :param expected_metapaths: a list of expected metapaths
    :return: a list of python dictionaries, consisting of metapath-based neighbor pairs and intermediate paths
    """
    outs = []
    for metapath in expected_metapaths:  # 遍历三条元路径如(0,1,0)
        # consider only the edges relevant to the expected metapath
        mask = np.zeros(M.shape, dtype=bool)  # 邻接矩阵size
        for i in range((len(metapath) - 1) // 2):  # 只看元路径的一半
            temp = np.zeros(M.shape, dtype=bool)
            temp[np.ix_(type_mask == metapath[i], type_mask == metapath[i + 1])] = True  # 所有ap pa都为true
            temp[np.ix_(type_mask == metapath[i + 1], type_mask == metapath[i])] = True
            mask = np.logical_or(mask, temp)  
        partial_g_nx = nx.from_numpy_matrix((M * mask).astype(int))  # 只挑出a和p的连边构图

        # only need to consider the former half of the metapath
        # e.g., we only need to consider 0-1-2 for the metapath 0-1-2-1-0
        metapath_to_target = {}
        for source in (type_mask == metapath[0]).nonzero()[0]:  # 所有type为0的，从0排序
            for target in (type_mask == metapath[(len(metapath) - 1) // 2]).nonzero()[0]:  # 所有type为2的，从0排序
                # check if there is a possible valid path from source to target node
                has_path = False
                single_source_paths = nx.single_source_shortest_path(  # 单源最短路径
                    partial_g_nx, source, cutoff=(len(metapath) + 1) // 2 - 1)  # 停止搜索的长度 以能到达的target为key的节点
                if target in single_source_paths:  # 有路径到达target
                    has_path = True

                #if nx.has_path(partial_g_nx, source, target):
                if has_path:
                    shortests = [p for p in nx.all_shortest_paths(partial_g_nx, source, target) if
                                 len(p) == (len(metapath) + 1) // 2]  # 所有最短路径
                    if len(shortests) > 0:  # 至少存在一条
                        metapath_to_target[target] = metapath_to_target.get(target, []) + shortests  # 目标节点为key
        metapath_neighbor_paris = {}
        for key, value in metapath_to_target.items():  # 基于某条元路径，target只有一种类型，到达每一个target的所有实例
            for p1 in value:  # 任取两个实例，拼到一起 可能有正反向重复的
                for p2 in value:
                    metapath_neighbor_paris[(p1[0], p2[0])] = metapath_neighbor_paris.get((p1[0], p2[0]), []) + [
                        p1 + p2[-2::-1]]  # (u,u)
        outs.append(metapath_neighbor_paris)  # 三个字典
    return outs


def get_networkx_graph(neighbor_pairs, type_mask, ctr_ntype):  # 只有a类型节点
    indices = np.where(type_mask == ctr_ntype)[0]  # a类型节点的索引
    idx_mapping = {}
    for i, idx in enumerate(indices):  # 映射到从0开始 才能构图
        idx_mapping[idx] = i  # 相同类型节点，只用一个mapping
    G_list = []
    for metapaths in neighbor_pairs:  # 三条元路径各自构图
        edge_count = 0
        sorted_metapaths = sorted(metapaths.items())
        G = nx.MultiDiGraph()  # 多重图
        G.add_nodes_from(range(len(indices)))
        for (src, dst), paths in sorted_metapaths:  # 认为是有向的  每两个u 经过采样有很多条路径要重复加边
            for _ in range(len(paths)):
                G.add_edge(idx_mapping[src], idx_mapping[dst])  # 头节点和尾节点作为边，重复加边，邻居重复
                edge_count += 1
        G_list.append(G)
    return G_list


def get_edge_metapath_idx_array(neighbor_pairs):
    all_edge_metapath_idx_array = []
    for metapath_neighbor_pairs in neighbor_pairs:  # 字典
        sorted_metapath_neighbor_pairs = sorted(metapath_neighbor_pairs.items())  # 排序,从0开始
        edge_metapath_idx_array = []
        for _, paths in sorted_metapath_neighbor_pairs:  # 某一条元路径所有实例
            edge_metapath_idx_array.extend(paths)
        edge_metapath_idx_array = np.array(edge_metapath_idx_array, dtype=int)
        all_edge_metapath_idx_array.append(edge_metapath_idx_array)   # 列表，每个a按顺序排列，索引到列表是一个元路径实例矩阵
        # print(edge_metapath_idx_array.shape)
    return all_edge_metapath_idx_array  # 把每个字典内容变成array 保留原来的indice
