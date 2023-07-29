
import copy
import numpy as np
import networkx as nx
import torch
import itertools
import scipy.sparse as sparse
from scipy.sparse import csr_matrix,coo_matrix


def threshold(x,num=0.8):
    y=copy.deepcopy(x)
    for i in range(len(y)):
        if y[i] >  num:
            y[i] = 1
        else:
            y[i] = 0
    return y

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def npz_pre(npz):
    npz = npz.log1p()
    npz.setdiag(1)
    npz.eliminate_zeros()
    return npz

def get_new(npz):
    npz = npz.tocoo()
    data = npz.data[npz.data == 1]
    row = npz.row[npz.data == 1]
    col = npz.col[npz.data == 1]
    new_npz =  coo_matrix(((data),(row,col)), shape = (npz.shape[0],npz.shape[0]))
    # new = npz.toarray()
    # new[new<1]=0
    # new_npz = coo_matrix(new)
    # new_npz.eliminate_zeros()
    return new_npz

def get_new_dynamic(old_npz_list, npz):
    sum_all = sum(old_npz_list)
    sum_all.data = np.power(sum_all.data,0)
    temp = npz.power(0) - sum_all 
    return get_new(temp)


def get_edge(npz):
    tensor_temp = torch.tensor(np.array(npz.nonzero()),dtype=torch.long)
    return tensor_temp

def get_edge_list(npz_list):
    list_edge = []
    for i in range(len(npz_list)):
        tensor_temp = torch.tensor(np.array(npz_list[i].nonzero()),dtype=torch.long)
        list_edge.append(tensor_temp)
    return list_edge


def get_edge_weight(npz):
    npz = npz.tolil()
    tensor_temp = []
    for i in range(npz.shape[0]):
        if len(npz[i].data[0])!=0:
            tensor_temp.append(np.array(npz[i].data[0])/max(npz[i].data[0]))
        else:
            tensor_temp.append(np.array(1))
    return tensor_temp

def get_hyperedge(npz):
    npz = npz.tolil()
    tensor_temp = []
    for i in range(npz.shape[0]):
        if len(npz[i].data[0])!=0:
            tensor_temp.append(list(npz[i].nonzero()[1]))
    return tensor_temp

def get_hyperedge_weight(npz):
    npz = npz.tolil()
    tensor_temp = []
    for i in range(npz.shape[0]):
        if len(npz[i].data[0])!=0:
            tensor_temp.extend(torch.tensor(np.array(npz[i].data[0])/max(npz[i].data[0]),dtype=torch.float))
        # else:
        #     tensor_temp.append(np.array(1))
    return torch.tensor(tensor_temp)

def get_link_labels(pos_edge_index, neg_edge_index,device):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

def new_link_labels(pos_edge_index,device):
    E = pos_edge_index.size(1) 
    link_labels = torch.ones(E, dtype=torch.float, device=device)
    return link_labels

def hier_edge(hg_subclass):
    edge_list = []
    for i in range(len(hg_subclass)):
        b = list(itertools.permutations(hg_subclass[i],2))
        edge_list.extend(b)
    edge_list = torch.tensor(edge_list).T
    return edge_list


def npz_threshold(npz):
    threshold =int(np.percentile(npz.data, (80), method='midpoint'))
    dat = npz.data[npz.data>=threshold]
    row = npz.row[npz.data>=threshold]
    col = npz.col[npz.data>=threshold]
    new_npz =  coo_matrix(((dat),(row,col)), shape = (npz.shape[0],npz.shape[0]))
    return new_npz
