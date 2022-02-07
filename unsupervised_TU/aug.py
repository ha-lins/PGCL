import os
import os.path as osp
import shutil

import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data

from itertools import repeat, product
import numpy as np

from copy import deepcopy
import pdb
import math
from scipy.linalg import expm
import time
import datetime


class TUDataset_aug(InMemoryDataset):
    r"""A variety of graph kernel benchmark datasets, *.e.g.* "IMDB-BINARY",
    "REDDIT-BINARY" or "PROTEINS", collected from the `TU Dortmund University
    <https://chrsmrrs.github.io/datasets>`_.
    In addition, this dataset wrapper provides `cleaned dataset versions
    <https://github.com/nd7141/graph_datasets>`_ as motivated by the
    `"Understanding Isomorphism Bias in Graph Data Sets"
    <https://arxiv.org/abs/1910.12091>`_ paper, containing only non-isomorphic
    graphs.

    .. note::
        Some datasets may not come with any node labels.
        You can then either make use of the argument :obj:`use_node_attr`
        to load additional continuous node attributes (if present) or provide
        synthetic node features using transforms such as
        like :class:`torch_geometric.transforms.Constant` or
        :class:`torch_geometric.transforms.OneHotDegree`.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The `name
            <https://chrsmrrs.github.io/datasets/docs/datasets/>`_ of the
            dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        use_node_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous node attributes (if present).
            (default: :obj:`False`)
        use_edge_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous edge attributes (if present).
            (default: :obj:`False`)
        cleaned: (bool, optional): If :obj:`True`, the dataset will
            contain only non-isomorphic graphs. (default: :obj:`False`)
    """

    url = ('http://ls11-www.cs.tu-dortmund.de/people/morris/'
           'graphkerneldatasets')
    cleaned_url = ('https://raw.githubusercontent.com/nd7141/'
                   'graph_datasets/master/datasets')

    def __init__(self, root, name, transform=None, pre_transform=None,
                 pre_filter=None, use_node_attr=False, use_edge_attr=False,
                 cleaned=False, aug=None, stro_aug=None, weak_aug2=None):
        self.name = name
        self.cleaned = cleaned
        super(TUDataset_aug, self).__init__(root, transform, pre_transform,
                                            pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # self.data.graph_idx = 0
        if self.data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            self.data.x = self.data.x[:, num_node_attributes:]
        if self.data.edge_attr is not None and not use_edge_attr:
            num_edge_attributes = self.num_edge_attributes
            self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]
        if not (self.name == 'MUTAG' or self.name == 'PTC_MR' or self.name == 'DD' or self.name == 'PROTEINS' or self.name == 'NCI1' or self.name == 'NCI109'):
            edge_index = self.data.edge_index[0, :].numpy()
            _, num_edge = self.data.edge_index.size()
            nlist = [edge_index[n] + 1 for n in range(num_edge - 1) if edge_index[n] > edge_index[n + 1]]
            nlist.append(edge_index[-1] + 1)

            num_node = np.array(nlist).sum()
            self.data.x = torch.ones((num_node, 1))

            edge_slice = [0]
            k = 0
            for n in nlist:
                k = k + n
                edge_slice.append(k)
            self.slices['x'] = torch.tensor(edge_slice)

            '''
            print(self.data.x.size())
            print(self.slices['x'])
            print(self.slices['x'].size())
            assert False
            '''

        self.aug = aug
        self.stro_aug = stro_aug
        self.weak_aug2 = weak_aug2

    @property
    def raw_dir(self):
        name = 'raw{}'.format('_cleaned' if self.cleaned else '')
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self):
        name = 'processed{}'.format('_cleaned' if self.cleaned else '')
        return osp.join(self.root, self.name, name)

    @property
    def num_node_labels(self):
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_labels(self):
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0

    @property
    def num_edge_attributes(self):
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1) - self.num_edge_labels

    @property
    def raw_file_names(self):
        names = ['A', 'graph_indicator']
        return ['{}_{}.txt'.format(self.name, name) for name in names]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        url = self.cleaned_url if self.cleaned else self.url
        folder = osp.join(self.root, self.name)
        print(url, self.name)
        path = download_url('{}/{}.zip'.format(url, self.name), folder)
        extract_zip(path, folder)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(folder, self.name), self.raw_dir)

    def process(self):
        self.data, self.slices = read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.pre_get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))

    def get_num_feature(self):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[0]

        for key in self.data.keys:
            try:
                item, slices = self.data[key], self.slices[key]
            except:
                break
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key,
                                        item)] = slice(slices[0],
                                                       slices[0 + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        _, num_feature = data.x.size()

        return num_feature

    def pre_get(self, idx):
        data = self.data.__class__()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key,
                    item)] = slice(slices[idx],
                    slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        return data

    def get(self, idx):
        data = self.data.__class__()
        # data.graph_idx = self.data.graph_idx
        # self.data.graph_idx += 1

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key,
                    item)] = slice(slices[idx],
                    slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        """
        edge_index = data.edge_index
        node_num = data.x.size()[0]
        edge_num = data.edge_index.size()[1]
        data.edge_index = torch.tensor([[edge_index[0, n], edge_index[1, n]] for n in range(edge_num) if edge_index[0, n] < node_num and edge_index[1, n] < node_num] + [[n, n] for n in range(node_num)], dtype=torch.int64).t()
        """

        node_num = data.edge_index.max()
        sl = torch.tensor([[n,n] for n in range(node_num)]).t()
        data.edge_index = torch.cat((data.edge_index, sl), dim=1)


        if self.aug == 'dnodes':
            data_aug = drop_nodes(deepcopy(data))
        elif self.aug == 'dedge_nodes':
            data_aug = drop_edge_nodes(deepcopy(data))
        elif self.aug == 'pedges':
            data_aug = permute_edges(deepcopy(data))
        elif self.aug == 'subgraph':
            data_aug = subgraph(deepcopy(data))
        elif self.aug == 'mask_nodes':
            data_aug = mask_nodes(deepcopy(data))
        elif self.aug == 'diff':
            data_aug = ppr_aug(deepcopy(data))
        elif self.aug == 'rotate':
            data_aug = rotate(deepcopy(data))
        elif self.aug == 'clip':
            data_aug = clip(deepcopy(data))
        elif self.aug == 'none':
            data_aug = deepcopy(data)
            data_aug.x = torch.ones((data.edge_index.max()+1, 1))
        elif self.aug == 'random2':
            n = np.random.randint(2)
            if n == 0:
                data_aug = drop_nodes(deepcopy(data))
            elif n == 1:
                data_aug = subgraph(deepcopy(data))
            else:
                print('sample error')
                assert False

        elif self.aug == 'random3':
            n = np.random.randint(3)
            if n == 0:
                data_aug = drop_nodes(deepcopy(data))
            elif n == 1:
                data_aug = permute_edges(deepcopy(data))
            elif n == 2:
                data_aug = subgraph(deepcopy(data))
            else:
                print('sample error')
                assert False

        elif self.aug == 'random4':
            n = np.random.randint(4)
            if n == 0:
                data_aug = drop_nodes(deepcopy(data))
            elif n == 1:
                data_aug = permute_edges(deepcopy(data))
            elif n == 2:
                data_aug = subgraph(deepcopy(data))
            elif n == 3:
                data_aug = mask_nodes(deepcopy(data))
            else:
                print('sample error')
                assert False
        else:
            print('augmentation error')
            assert False


        if self.weak_aug2 == 'dnodes':
            data_weak_aug2 = drop_nodes(deepcopy(data))
        elif self.weak_aug2 == 'dedge_nodes':
            data_weak_aug2 = drop_edge_nodes(deepcopy(data))
        elif self.weak_aug2 == 'pedges':
            data_weak_aug2 = permute_edges(deepcopy(data))
        elif self.weak_aug2 == 'subgraph':
            data_weak_aug2 = subgraph(deepcopy(data))
        elif self.weak_aug2 == 'mask_nodes':
            data_weak_aug2 = mask_nodes(deepcopy(data))
        elif self.aug == 'rotate':
            data_weak_aug2 = rotate(deepcopy(data))
        elif self.aug == 'clip':
            data_weak_aug2 = clip(deepcopy(data))
        elif self.weak_aug2 == 'diff':
            data_weak_aug2 = ppr_aug(deepcopy(data))
        elif self.weak_aug2 == 'none':
            """
            if data.edge_index.max() > data.x.size()[0]:
                print(data.edge_index)
                print(data.x.size())
                assert False
            """
            data_weak_aug2 = deepcopy(data)
            data_weak_aug2.x = torch.ones((data.edge_index.max()+1, 1))

        # print(self.stro_aug)
        if self.stro_aug == 'stro_subgraph':
            data_stro_aug = stro_subgraph(deepcopy(data))
        elif self.stro_aug == 'stro_dnodes':
            data_stro_aug = stro_drop_nodes(deepcopy(data))
        elif self.stro_aug == 'subgraph':
            data_stro_aug = subgraph(deepcopy(data))
        elif self.stro_aug == None or self.stro_aug == 'none':
            """
            if data.edge_index.max() > data.x.size()[0]:
                print(data.edge_index)
                print(data.x.size())
                assert False
            """
            data_stro_aug = deepcopy(data)
            data_stro_aug.x = torch.ones((data.edge_index.max()+1, 1))
        else:
            print('stro_subgraph augmentation error')
            assert False

        if self.weak_aug2 != None:
            return data, data_aug, data_weak_aug2
        return data, data_aug, data_stro_aug

    # def get(self, idx):
    #     data = self.data.__class__()
    #
    #     if hasattr(self.data, '__num_nodes__'):
    #         data.num_nodes = self.data.__num_nodes__[idx]
    #
    #     for key in self.data.keys:
    #         item, slices = self.data[key], self.slices[key]
    #         if torch.is_tensor(item):
    #             s = list(repeat(slice(None), item.dim()))
    #             s[self.data.__cat_dim__(key,
    #                                     item)] = slice(slices[idx],
    #                                                    slices[idx + 1])
    #         else:
    #             s = slice(slices[idx], slices[idx + 1])
    #         data[key] = item[s]
    #
    #     """
    #     edge_index = data.edge_index
    #     node_num = data.x.size()[0]
    #     edge_num = data.edge_index. size()[1]
    #     data.edge_index = torch.tensor([[edge_index[0, n], edge_index[1, n]] for n in range(edge_num) if edge_index[0, n] < node_num and edge_index[1, n] < node_num] + [[n, n] for n in range(node_num)], dtype=torch.int64).t()
    #     """
    #
    #     node_num = data.edge_index.max()
    #     sl = torch.tensor([[n,n] for n in range(node_num)]).t()
    #     data.edge_index = torch.cat((data.edge_index, sl), dim=1)
    #
    #     data_first_left = permute_edges(deepcopy(data))
    #     data_first_right = drop_edge_nodes(deepcopy(data))
    #     data_sec_left_left = permute_edges(deepcopy(data_first_left))
    #     data_sec_left_right = drop_edge_nodes(deepcopy(data_first_left))
    #     data_sec_right_left = permute_edges(deepcopy(data_first_right))
    #     data_sec_right_right = drop_edge_nodes(deepcopy(data_first_right))
    #
    #     data_first_left = permute_edges(deepcopy(data))
    #     data_first_right = drop_nodes(deepcopy(data_first_left))
    #     data_sec_left_left = permute_edges(deepcopy(data_first_left))
    #     data_sec_left_right = drop_nodes(deepcopy(data_sec_left_left))
    #     data_sec_right_left = permute_edges(deepcopy(data_first_right))
    #     data_sec_right_right = drop_nodes(deepcopy(data_sec_right_left))

        # if self.weak_aug2 != None:
        #     return data, data_aug, data_weak_aug2, data_stro_aug
        # return data, data_first_left, data_first_right, data_sec_left_left, data_sec_left_right, data_sec_right_left, \
        #        data_sec_right_right
        # return data

def get_adj_matrix(data) -> np.ndarray:
    num_nodes = data.x.shape[0]
    adj_matrix = np.zeros(shape=(num_nodes, num_nodes))
    for i, j in zip(data.edge_index[0], data.edge_index[1]):
        adj_matrix[i, j] = 1.
    return adj_matrix

def get_ppr_matrix(
        adj_matrix: np.ndarray,
        alpha: float = 0.1) -> np.ndarray:
    num_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix + np.eye(num_nodes)
    D_tilde = np.diag(1/np.sqrt(A_tilde.sum(axis=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return alpha * np.linalg.inv(np.eye(num_nodes) - (1 - alpha) * H)

def get_top_k_matrix(A: np.ndarray, k: int = 128) -> np.ndarray:
    num_nodes = A.shape[0]
    row_idx = np.arange(num_nodes)
    A[A.argsort(axis=0)[:num_nodes - k], row_idx] = 0.
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1 # avoid dividing by zero
    return A/norm


def get_clipped_matrix(A: np.ndarray, eps: float = 0.01) -> np.ndarray:
    num_nodes = A.shape[0]
    A[A < eps] = 0.
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1 # avoid dividing by zero
    return A/norm

def ppr_aug(data):
    adj_matrix = get_adj_matrix(data)
    # obtain exact PPR matrix
    ppr_matrix = get_ppr_matrix(adj_matrix)
    # print(ppr_matrix)

    k = 128
    eps = None

    if k:
        # print(f'Selecting top {k} edges per node.')
        ppr_matrix = get_top_k_matrix(ppr_matrix, k=k)
    elif eps:
        print(f'Selecting edges with weight greater than {eps}.')
        ppr_matrix = get_clipped_matrix(ppr_matrix, eps=eps)
    else:
        raise ValueError

    edge_index = torch.tensor(ppr_matrix).nonzero().t()
    data.edge_index = edge_index
    # print(data.edge_index)
    return data

def drop_edge_nodes(data):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    ratio = 0.1
    degrees = {}
    for n in data.edge_index[0]:
        if n.item() in degrees:
            degrees[n.item()] += 1
        else:
            degrees[n.item()] = 1
    degrees = sorted(degrees.items(), key = lambda item: item[1])

    idx_drop = np.array([n[0] for n in degrees[:int(len(degrees) * ratio)]])
    drop_num = len(idx_drop)
    drop_ratio = drop_num / node_num
    # print('drop_ratio is: {}'.format(drop_ratio))

    idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
    edge_idx_drop = [idx for idx, n in enumerate(data.edge_index[0]) if n.item() in idx_drop or data.edge_index[1][idx].item() in idx_drop]
    edge_idx_nondrop = [n for n in range(len(data.edge_index[0])) if not n in edge_idx_drop]

    idx_dict = {idx_nondrop[n]:n for n in list(range(node_num - drop_num))}

    edge_index = data.edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    data.edge_index = edge_index
    try:
        data.edge_attr = data.edge_attr[edge_idx_nondrop]
    except:
        pass
    return data

# def flip(x, dim):
#     print(x)
#     indices = [slice(None)] * x.dim()
#     indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
#                                 dtype=torch.long, device=x.device)
#     return x[tuple(indices)]

def rotate(data):
    data.x = torch.flip(data.x, [1])
    data.edge_index = torch.flip(data.edge_index, [1])

    return data


def clip(data):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()

    drop_num = int(node_num / 10)     # ratio for remained nodes

    # idx_drop = np.random.choice(node_num, drop_num, replace=False)
    idx_drop = np.arange(drop_num)

    idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
    idx_dict = {idx_nondrop[n]:n for n in list(range(node_num - drop_num))}

    edge_index = data.edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    data.edge_index = edge_index
    return data

def drop_nodes(data):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()

    drop_num = int(node_num / 10)     # ratio for remained nodes
    # print('[info] drop_num is:{}'.format(drop_num))

    idx_drop = np.random.choice(node_num, drop_num, replace=False)
    # print('[info] idx_drop is:{}'.format(idx_drop))

    idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
    idx_dict = {idx_nondrop[n]:n for n in list(range(node_num - drop_num))}

    # data.x = data.x[idx_nondrop]
    edge_index = data.edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    data.edge_index = edge_index

    return data

def stro_drop_nodes(data):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()

    degrees = {}
    for n in data.edge_index[0]:
        if n.item() in degrees:
            degrees[n.item()] += 1
        else:
            degrees[n.item()] = 1
    for key, val in degrees.items():
        degrees[key] = math.log(val)

    min_d, max_d = min(degrees.values()), max(degrees.values())

    idx_drop = np.array([n for n in degrees if (degrees[n] - min_d) / (max_d - min_d) <= 0.2])

    drop_num = len(idx_drop)
    drop_ratio = len(idx_drop) / node_num
    # print('drop_ratio is: {}'.format(drop_ratio))

    idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
    idx_dict = {idx_nondrop[n]:n for n in list(range(node_num - drop_num))}

    # data.x = data.x[idx_nondrop]
    edge_index = data.edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    data.edge_index = edge_index
    # print('[info] stro edge_index is:{}'.format(len(edge_index[0])))

    # edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    # edge_index = [[edge_index[0, n], edge_index[1, n]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)] + [[n, n] for n in idx_nondrop]
    # data.edge_index = torch.tensor(edge_index).transpose_(0, 1)

    return data


def permute_edges(data):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num / 10)

    edge_index = data.edge_index.transpose(0, 1).numpy()

    idx_add = np.random.choice(node_num, (permute_num, 2))
    # idx_add = [[idx_add[0, n], idx_add[1, n]] for n in range(permute_num) if not (idx_add[0, n], idx_add[1, n]) in edge_index]

    # edge_index = np.concatenate((np.array([edge_index[n] for n in range(edge_num) if not n in np.random.choice(edge_num, permute_num, replace=False)]), idx_add), axis=0)
    # edge_index = np.concatenate((edge_index[np.random.choice(edge_num, edge_num-permute_num, replace=False)], idx_add), axis=0)
    rand_ids = np.random.choice(edge_num, edge_num-permute_num, replace=False)
    edge_index = edge_index[rand_ids]
    # edge_index = [edge_index[n] for n in range(edge_num) if not n in np.random.choice(edge_num, permute_num, replace=False)] + idx_add
    data.edge_index = torch.tensor(edge_index).transpose_(0, 1)
    try:
        data.edge_attr = data.edge_attr[edge_idx_nondrop]
    except:
        pass

    return data

# def permute_edges(data):
#
#     node_num, _ = data.x.size()
#     _, edge_num = data.edge_index.size()
#     permute_num = int(edge_num / 10)
#
#     edge_index = data.edge_index.transpose(0, 1).numpy()
#
#     idx_add = np.random.choice(node_num, (permute_num, 2))
#     # idx_add = [[idx_add[0, n], idx_add[1, n]] for n in range(permute_num) if not (idx_add[0, n], idx_add[1, n]) in edge_index]
#
#     # edge_index = np.concatenate((np.array([edge_index[n] for n in range(edge_num) if not n in np.random.choice(edge_num, permute_num, replace=False)]), idx_add), axis=0)
#     # edge_index = np.concatenate((edge_index[np.random.choice(edge_num, edge_num-permute_num, replace=False)], idx_add), axis=0)
#     edge_index = edge_index[np.random.choice(edge_num, edge_num-permute_num, replace=False)]
#     # edge_index = [edge_index[n] for n in range(edge_num) if not n in np.random.choice(edge_num, permute_num, replace=False)] + idx_add
#     data.edge_index = torch.tensor(edge_index).transpose_(0, 1)
#
#     return data

def subgraph(data):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * 0.2)

    edge_index = data.edge_index.numpy()

    idx_sub = [np.random.randint(node_num, size=1)[0]]
    idx_neigh = set([n for n in edge_index[1][edge_index[0]==idx_sub[0]]])

    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        if count > node_num:
            break
        if len(idx_neigh) == 0:
            break
        sample_node = np.random.choice(list(idx_neigh))
        if sample_node in idx_sub:
            continue
        idx_sub.append(sample_node)
        idx_neigh.union(set([n for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))

    idx_drop = [n for n in range(node_num) if not n in idx_sub]
    idx_nondrop = idx_sub
    idx_dict = {idx_nondrop[n]:n for n in list(range(len(idx_nondrop)))}

    # data.x = data.x[idx_nondrop]
    edge_index = data.edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    data.edge_index = edge_index

    # edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    # edge_index = [[edge_index[0, n], edge_index[1, n]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)] + [[n, n] for n in idx_nondrop]
    # data.edge_index = torch.tensor(edge_index).transpose_(0, 1)

    return data

def stro_subgraph(data):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * 0.1)
    # print('[info] stro sub_num is:{}'.format(sub_num))
    edge_index = data.edge_index.numpy()

    idx_sub = [np.random.randint(node_num, size=1)[0]]
    idx_neigh = set([n for n in edge_index[1][edge_index[0]==idx_sub[0]]])

    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        # if count > node_num:
        #     break
        if len(idx_neigh) == 0:
            break
        sample_node = np.random.choice(list(idx_neigh))
        # print('[info] stro idx_neigh is:{}'.format(idx_neigh))
        # print('[info] stro sample_node is:{}'.format(sample_node))

        if sample_node in idx_sub:
            continue
        idx_sub.append(sample_node)
        idx_neigh.union(set([n for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))

    idx_drop = [n for n in range(node_num) if not n in idx_sub]
    # print('[info] stro idx_drop is:{}'.format(idx_drop))
    idx_nondrop = idx_sub
    idx_dict = {idx_nondrop[n]:n for n in list(range(len(idx_nondrop)))}
    # print('[info] stro idx_sub is:{}'.format(len(idx_sub)))

    # data.x = data.x[idx_nondrop]
    edge_index = data.edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_nondrop, :] = 0
    adj[:, idx_nondrop] = 0
    edge_index = adj.nonzero().t()
    # print('[info] stro edge_index is:{}'.format(len(edge_index[0])))

    data.edge_index = edge_index

    # edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    # edge_index = [[edge_index[0, n], edge_index[1, n]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)] + [[n, n] for n in idx_nondrop]
    # data.edge_index = torch.tensor(edge_index).transpose_(0, 1)

    return data

def mask_nodes(data):

    node_num, feat_dim = data.x.size()
    mask_num = int(node_num / 10)

    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = torch.tensor(np.random.normal(loc=0.5, scale=0.5, size=(mask_num, feat_dim)), dtype=torch.float32)

    return data

