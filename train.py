import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
from copy import deepcopy
import networkx as nx
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from dhg import Graph, Hypergraph
from dhg.data import Cora, Pubmed, Citeseer
from dhg.models import HGNN, HGNNP, HNHN
from dhg.random import set_seed
# from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator
from sklearn.metrics import roc_auc_score, average_precision_score,f1_score,precision_score,recall_score, accuracy_score
import numpy as np
import csv
import pandas as pd
import json
import scipy.sparse as sparse
from scipy.sparse import csr_matrix,coo_matrix
import scipy.sparse as sp
import itertools
from sklearn.metrics import roc_auc_score, average_precision_score,f1_score,precision_score,recall_score
from torch_geometric.utils import negative_sampling
from utils import threshold, normalization, npz_pre, get_new, get_edge, get_link_labels,hier_edge , get_hyperedge, get_hyperedge_weight, npz_threshold,new_link_labels,group_select
from models import HGNN_link, HGNNP_link, GCN_link, HyperGCN_link, GraphSAGE_link,GIN_link,GAT_link


set_seed(2022)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
hidden_dim = 128
out_dim = 16

data_train = 'subclass'

if data_train == 'subclass':
    lambd = 0.7
    e = 0.05
if data_train == 'group':
    lambd = 0.5
    e = 0.1
if data_train == 'mag_cs':
    lambd = 0.5
    e = 0.2

print('This result is based on', data_train,' data')

def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(net, X, HG, train_edge,train_edge_new, optimizer, epoch):
    net.train()
    st = time.time()
    optimizer.zero_grad()
    logit, logit_new ,a  = net(X, HG, train_edge, train_edge_new)
    #logit = logit[0]
    link_label = get_link_labels(train_pos_edge1,train_neg_edge,device)
    link_newlabel = new_link_labels( train_edge_new ,device)
    loss1 = F.binary_cross_entropy_with_logits(logit, link_label).to(device)
    loss2 = F.binary_cross_entropy_with_logits(logit_new, link_newlabel).to(device)
    loss = lambd*loss1 + (1-lambd)*loss2
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
    return loss.item(), a
    

@torch.no_grad()
def infer(net, X, HG, edge , edge_new, epoch, test=False):
    net.eval()
    logit, logit_new, a = net(X, HG, edge, edge_new)

    link_newlabel = np.ones(len(logit_new))
    if not test:
        link_label = get_link_labels(valid_pos_edge, valid_neg_edge, device)
        loss_valid = F.binary_cross_entropy_with_logits(logit, link_label)

        logit = logit.cpu().detach().numpy()

        logit_new = logit_new.cpu().detach().numpy()
        link_label = link_label.cpu().detach().numpy()
        num = np.percentile(logit,66.6)
        logit = threshold(logit,num = num)
        logit_new = threshold(logit_new,num = num)

        res_new = np.sum(logit_new)/np.sum(link_newlabel)
        auc = roc_auc_score(link_label, logit)
        acc = accuracy_score(link_label, logit)
        ap = average_precision_score(link_label, logit)
        f1 = f1_score(link_label, logit)
        precision = precision_score(link_label, logit)
        recall = recall_score(link_label, logit)
        return auc,acc, ap, f1, precision, recall, res_new,loss_valid
        #res_new = roc_auc_score(logit_new, link_newlabel)
    else:
        link_label = get_link_labels(test_pos_edge, test_neg_edge, device)
        link_label = link_label.cpu().detach().numpy()
        logit = logit.cpu().detach().numpy()

        logit_new = logit_new.cpu().detach().numpy()
        num = np.percentile(logit,66.6)
        logit = threshold(logit,num = num)
        logit_new = threshold(logit_new,num = num)

        auc = roc_auc_score(link_label, logit)
        acc = accuracy_score(link_label, logit)
        res_new = np.sum(logit_new)/np.sum(link_newlabel)
        ap = average_precision_score(link_label, logit)
        f1 = f1_score(link_label, logit)
        precision = precision_score(link_label, logit)
        recall = recall_score(link_label, logit)
        #res_new = roc_auc_score(logit_new, link_newlabel)
        return auc,acc, ap, f1, precision, recall, res_new



if data_train == 'subclass':
    train_npz_co_list = []
    for i in range(2000,2020):
        temp_co = sparse.load_npz('./Data/co_class/subclass/'+str(i)+'.npz')
        temp_co = npz_threshold(temp_co)
        train_npz_co_list.append(npz_pre(temp_co))
    train_npz_co_graph = sum(train_npz_co_list[0:-1])
    train_npz_co = train_npz_co_list[-1]


    valid_npz = sparse.load_npz('./Data/co_class/subclass/2020.npz')
    valid_npz = npz_threshold(valid_npz)
    test_npz = sparse.load_npz('./Data/co_class/subclass/2021.npz')
    test_npz = npz_threshold(test_npz)
    train_npz_ci_list = []
    for i in range(2000,2019):
        temp_ci = sparse.load_npz('./Data/citation_merge/subclass/'+str(i)+'.npz')
        temp_ci = npz_threshold(temp_ci)
        train_npz_ci_list.append(temp_ci)
    train_npz_ci = sum(train_npz_ci_list)
    filename3 = './Data/feature/subclass_feature.json'
    filename = './Data/hierarchical_hg/hier_subclass.json'
    with open(filename,'r',encoding='utf-8') as f:
        hg_subclass = json.load(f)
    with open(filename3,'r',encoding='utf-8') as f:
        feature_subclass = json.load(f)

elif data_train=='group':
    train_npz_co_list = []
    for i in range(2000,2020):
        temp_co = sparse.load_npz('./Data_22/co_class/group/'+str(i)+'.npz')
        train_npz_co_list.append(temp_co)
    train_npz_co_graph = sum(train_npz_co_list[0:-1])
    train_npz_co = train_npz_co_list[-1]

    valid_npz = sparse.load_npz('./Data/co_class/group/2020.npz')
    test_npz = sparse.load_npz('.Data/co_class/group/2021.npz')
    train_npz_ci_list = []
    for i in range(2000,2019):
        temp_ci = sparse.load_npz('./Data/citation_merge/group/'str(i)+'.npz')
        train_npz_ci_list.append(temp_ci)
    train_npz_ci = sum(train_npz_ci_list)

    filename3 = './Data/feature/group_feature1.json'
    filename = './Data/hierarchical_hg/hier_group1.json'
    with open(filename,'r',encoding='utf-8') as f:
        hg_subclass = json.load(f)

    with open(filename3,'r',encoding='utf-8') as f:
        feature_subclass = json.load(f)


elif data_train =='mag_cs':
    train_npz_co_list = []
    for i in range(2000,2016):
        temp_co = sparse.load_npz('./MAG_data/coedge/'+str(i)+'.npz')
        temp_co = npz_threshold(temp_co)
        train_npz_co_list.append(npz_pre(temp_co))
    train_npz_co_graph = sum(train_npz_co_list[0:-1])
    train_npz_co = train_npz_co_list[-1]
    valid_npz = sparse.load_npz('./MAG_data/coedge/2016.npz')
    valid_npz = npz_threshold(valid_npz)
    test_npz = sparse.load_npz('./MAG_data/coedge/2017.npz')
    test_npz = npz_threshold(test_npz)
    train_npz_ci_list = []
    for i in range(2000,2015):
        temp_ci = sparse.load_npz('./MAG_data/cita_edge/'+str(i)+'.npz')
        temp_ci = npz_threshold(temp_ci)
        train_npz_ci_list.append(temp_ci)
    train_npz_ci = sum(train_npz_ci_list)

    filename3 = './MAG_data/paper_field/field5_emb.json'
    filename = './MAG_data/paper_field/field_cluster.json'
    hg_subclass = []
    for line in  open(filename, "r", encoding='utf-8'):
        hg_subclass.append(json.loads(line))
    feature_subclass = []
    for line in  open(filename3, "r", encoding='utf-8'):
        feature_subclass.append(json.loads(line))

train_npz_co_graph = npz_pre(train_npz_co_graph)

train_npz_graph_ = train_npz_co_graph.power(0)
train_npz_ = train_npz_co.power(0) 
valid_npz_ = valid_npz.power(0)
test_npz_ = test_npz.power(0)
train_newnpz = train_npz_ - train_npz_graph_
train_newnpz = get_new(train_newnpz)
valid_newnpz = valid_npz_ - train_npz_
valid_newnpz = get_new(valid_newnpz)
test_newnpz = test_npz_ - train_npz_
test_newnpz = get_new(test_newnpz)



####要改成合成版本的
train_pos_edge1= get_edge(train_npz_co)
train_pos_edge2 = get_edge(train_npz_ci)
train_pos_edge3 = hier_edge(hg_subclass)
train_pos_edge_all = torch.cat([train_pos_edge1, train_pos_edge2, train_pos_edge3],dim =1)
train_pos_edge_graph1 = get_edge(train_npz_co_graph)
train_pos_edge_all_graph = torch.cat([train_pos_edge_graph1, train_pos_edge2, train_pos_edge3],dim =1)
################
train_neg_edge = negative_sampling(edge_index=train_pos_edge_all, num_nodes= train_npz_co.shape[0],
    num_neg_samples=train_pos_edge1.size(1)*2,force_undirected=True)
train_edge = torch.cat([train_pos_edge1, train_neg_edge],dim=-1)
train_newedge = get_edge(train_newnpz)

valid_pos_edge = get_edge(valid_npz)
valid_neg_edge = negative_sampling(edge_index=valid_pos_edge, num_nodes= valid_npz.shape[0],
    num_neg_samples=valid_pos_edge.size(1)*2,force_undirected=True)
valid_edge = torch.cat([valid_pos_edge, valid_neg_edge],dim=-1)
valid_newedge = get_edge(valid_newnpz)

test_pos_edge = get_edge(test_npz)
test_neg_edge = negative_sampling(edge_index=test_pos_edge, num_nodes= test_npz.shape[0],
    num_neg_samples=test_pos_edge.size(1)*2,force_undirected=True)
test_edge = torch.cat([test_pos_edge, test_neg_edge],dim=-1)
test_newedge = get_edge(test_newnpz)


##训练集特征
X = torch.Tensor(feature_subclass)
feature_dim = X.shape[1]

num_vertices = train_npz_co.get_shape()[0]
G1 = Graph(num_vertices, list(map(tuple,zip(*train_pos_edge_graph1.cpu().tolist()))))
G2 = Graph(num_vertices, list(map(tuple,zip(*train_pos_edge2.cpu().tolist()))))
G3 = Graph(num_vertices, list(map(tuple,zip(*train_pos_edge3.cpu().tolist()))))
G = Graph(num_vertices, list(map(tuple,zip(*train_pos_edge_all_graph.cpu().tolist()))))

HG = Hypergraph(num_v= num_vertices) 
HG.add_hyperedges(e_list = hg_subclass, group_name = 'hier') 
HG.add_hyperedges(e_list = get_hyperedge(train_npz_co_graph) , group_name = 'co-occurence') 
HG.add_hyperedges(e_list = get_hyperedge(train_npz_ci)  , group_name = 'citation') 

X = X.to(device)


from models import HyGAT_link,UniGAT,HyGAT_link1,HyperKCP, HyGAT_link3, UniGCN, UniGIN, UniSAGE

########### 
def run(X, G, HG, train_edge, valid_edge , valid_newedge,   test_edge, test_newedge, model_name = 'H3KCP'):


    net = H3KCP(feature_dim, 32, e, device)    

    Hyper= True



    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
    net = net.to(device)
    # init_network(net) 

    counter = 0
    min_loss = 100
    best_state = None
    best_epoch, best_val = -1, -1
    a_list = []

    for epoch in range(2000):
        # train
        if Hyper == True:
            temp,a= train(net, X, HG, train_edge,train_newedge, optimizer, epoch)
        else:
            temp,a = train(net, X, G, train_edge,train_newedge, optimizer, epoch)
        # validation
        a_list.append(a)
        if epoch % 1 == 0:
            with torch.no_grad():
                if Hyper == True:
                    val_auc, acc, ap, f1, precision, recall, res_new,loss_valid= infer(net, X, HG, valid_edge , valid_newedge,epoch, test=False)
                else:
                    val_auc, acc, ap, f1, precision, recall, res_new,loss_valid = infer(net, X, G, valid_edge , valid_newedge,epoch, test=False)
                
            if val_auc > best_val and loss_valid < min_loss:
                print(f"update best: {val_auc:.5f}")
                best_epoch = epoch
                best_val = val_auc
                min_loss = loss_valid
                best_state = deepcopy(net.state_dict())
                counter = 0
            else: 
                counter+=1
            if counter >= 200:
                print('early stop')
                break
    print("\ntrain finished!")
    print(f"best val: {best_val:.5f}")
    # test
    print("test...")
    net.load_state_dict(best_state)
    if Hyper == True:
        res = infer(net, X, HG, test_edge, test_newedge,epoch, test=True)
    else:
        res = infer(net, X, G, test_edge, test_newedge,epoch, test=True)
    print(f'model_name:', model_name)
    print(f"final result: epoch: {best_epoch}")
    print(res)
    print(sum(p.numel() for p  in net.parameters()))
    return 



run(X, G, HG, train_edge, valid_edge, valid_newedge, test_edge, test_newedge, model_name = 'H3KCP')
print('lambd:', lambd)
print('e:',e)
