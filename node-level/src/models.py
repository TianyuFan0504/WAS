import os
import pymetis
import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn.cluster import KMeans
import pyro
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv

from torch_geometric.utils import negative_sampling
from sklearn.neighbors import kneighbors_graph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GCN(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout_ratio, activation, norm_type="none"):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(GraphConv(input_dim, output_dim, activation=activation))
        else:
            self.layers.append(GraphConv(input_dim, hidden_dim, activation=activation))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(GraphConv(hidden_dim, hidden_dim, activation=activation))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))

            self.layers.append(GraphConv(hidden_dim, output_dim))

    def forward(self, g, feats):
        h = feats
        h_list = []

        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            h_list.append(h)
            if l != self.num_layers - 1:
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = self.dropout(h)
            
        return h_list, h


class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()
        if param['dataset'] == 'ogbn-arxiv':
            self.encoder = GCN(num_layers=param["num_layers"], input_dim=param["feat_dim"], hidden_dim=param["hidden_dim"], output_dim=param["label_dim"], dropout_ratio=param["dropout"], activation=F.relu, norm_type="batch").to(device)
        else:
            self.encoder = GCN(num_layers=param["num_layers"], input_dim=param["feat_dim"], hidden_dim=param["hidden_dim"], output_dim=param["label_dim"], dropout_ratio=param["dropout"], activation=F.relu).to(device)
        self.m = param['m']
   
        self.teacher_vector = nn.Linear(param['ssl_task_len'], param['label_dim'], bias=False)
        self.global_vector = nn.Linear(1, param['label_dim'], bias=False)
        self.teacher_vector_weight = self.teacher_vector.weight
        self.global_vector_weight = self.global_vector.weight
        self.teacher_select_vector = nn.Linear(param['ssl_task_len'], param['label_dim'], bias=False)
        self.global_select_vector = nn.Linear(1, param['label_dim'], bias=False)

        self.selector = torch.ones(1,param['num_teachers'])
        for param_q, param_k in zip(self.teacher_vector.parameters(), self.teacher_select_vector.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.global_vector.parameters(), self.global_select_vector.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        self.teacher_select_vector_weight = self.teacher_select_vector.weight
        self.global_select_vector_weight = self.global_select_vector.weight

        self.projS = nn.Linear(param['num_teachers'], param['num_teachers'], bias=False)
        

    def forward(self, g, feats):
        return self.encoder(g, feats)[1]

    def estimate_optimal_distribution(self, logits, logits_list, param):
            
        weight = torch.unsqueeze(logits, 2) * torch.unsqueeze(self.teacher_vector_weight, 0) * torch.unsqueeze(self.global_vector_weight, 0)  # N X D X S 
        weight_sum = weight.sum(dim=1)# B x T
        # weight_softmax = F.softmax(weight_sum, dim=-1)  # N X S 
        selector = torch.unsqueeze(logits, 2) * torch.unsqueeze(self.teacher_select_vector_weight, 0) * torch.unsqueeze(self.global_select_vector_weight, 0)  # N X D X S 
        selector_sum = selector.sum(dim=1)
        selector_sum = self.projS(selector_sum)
        selector_softmax = torch.softmax(selector_sum,dim=-1)
            
        wei = selector_softmax / torch.max(selector_softmax,dim = 1, keepdim=True)[0]
        select = pyro.distributions.RelaxedBernoulliStraightThrough(temperature = 1., probs=wei).rsample()# N x T
        # select = compensate_teachers(select)
        compensation = select.sum(dim=1).reshape((-1,1))
        
        while 0 in compensation:
            select = pyro.distributions.RelaxedBernoulliStraightThrough(temperature = 1., probs=wei).rsample()
            compensation = select.sum(dim=1).reshape((-1,1))

        assert 0 not in compensation, "ERROR in select"

        self.selector = (select.sum(dim = 0) / select.size(0)).detach()

        select_weight = pse_softmax(weight_sum , select)
        
        logits_list_softmax = F.softmax(logits_list/param['tau'], dim=-1)  # S X N X D
        
        logits_optimal = torch.unsqueeze(select_weight, 2) * logits_list_softmax.transpose(1, 0)  # N X S X D
        if param['dataset'] != 'cora' and param['dataset'] != 'citeseer' and param['dataset'] != 'pubmed':
            logits_optimal = (logits_optimal.sum(dim=1) + 1e-10) / (1 + param['label_dim'] * 1e-10)  # N X D
        else:
            logits_optimal = logits_optimal.sum(dim=1)  # N X D

        return logits_optimal.log()


            
    @torch.no_grad()
    def momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.teacher_vector.parameters(), self.teacher_select_vector.parameters()):
            param_k.data = param_k.data * (1. - self.m) + param_q.data * self.m

        for param_q, param_k in zip(self.global_vector.parameters(), self.global_select_vector.parameters()):
            param_k.data = param_k.data * (1. - self.m) + param_q.data * self.m

def pse_softmax(weight, quit):
    quit2 = quit.clone().detach()
    wq = weight * quit
    wq_exp = torch.exp(wq) * quit2
    w = wq_exp.sum(1, keepdim=True)
    output=(wq_exp/w).float()
    return output

class Par(nn.Module):

    def __init__(self, g, model, param):
        super(Par, self).__init__()
        self.dataset = param['dataset']
        if self.dataset in ['citeseer']:
            self.nparts = 1000
        elif self.dataset in ['amazon-photo', 'amazon-com']:
            self.nparts = 100
        else:
            self.nparts = 400

        self.pseudo_labels = self.get_label(g, self.nparts).to(device)
        self.disc = nn.Linear(param['hidden_dim'], self.nparts)
        self.sampled_indices = (self.pseudo_labels >= 0)

    def make_loss(self, embeddings):
        embeddings = self.disc(embeddings)
        output = F.log_softmax(embeddings, dim=1)
        loss = F.nll_loss(output[self.sampled_indices], self.pseudo_labels[self.sampled_indices])
        return loss

    def get_label(self, g, nparts):
        partition_file = '../models/saved/' + self.dataset + '_partition_%s.npy' % nparts

        if not os.path.exists(partition_file):
            print('Perform graph partitioning with Metis...')

            adj_coo = sp.coo_matrix(g.adjacency_matrix(transpose=True).to_dense().cpu().detach().numpy())
            node_num = adj_coo.shape[0]
            adj_list = [[] for _ in range(node_num)]
            for i, j in zip(adj_coo.row, adj_coo.col):
                if i == j:
                    continue
                adj_list[i].append(j)

            _, partition_labels =  pymetis.part_graph(adjacency=adj_list, nparts=nparts)
            np.save(partition_file, partition_labels)
            return torch.LongTensor(partition_labels)
        else:
            partition_labels = np.load(partition_file)
            return torch.LongTensor(partition_labels)


class Clu(nn.Module):

    def __init__(self, g, model, param):
        super(Clu, self).__init__()
        
        self.dataset = param['dataset']
        self.ncluster = 10

        self.pseudo_labels = self.get_label(g, self.ncluster).to(device)
        self.disc = nn.Linear(param['hidden_dim'], self.ncluster)
        self.sampled_indices = (self.pseudo_labels >= 0)

    def make_loss(self, embeddings):
        embeddings = self.disc(embeddings)
        output = F.log_softmax(embeddings, dim=1)
        loss = F.nll_loss(output[self.sampled_indices], self.pseudo_labels[self.sampled_indices])
        return loss

    def get_label(self, g, ncluster):
        cluster_file = '../models/saved/' + self.dataset + '_cluster_%s.npy' % ncluster

        if not os.path.exists(cluster_file):
            print('perform clustering with KMeans...')

            kmeans = KMeans(n_clusters=ncluster, random_state=0).fit(g.ndata["feat"].cpu())
            cluster_labels = kmeans.labels_

            np.save(cluster_file, cluster_labels)
            return torch.LongTensor(cluster_labels)
        else:
            cluster_labels = np.load(cluster_file)
            return torch.LongTensor(cluster_labels)


class DGI(nn.Module):
    def __init__(self, g, model, param):
        super(DGI, self).__init__()

        self.g = g
        self.gcn = model

        self.disc = Discriminator(param['hidden_dim'])
        self.b_xent = nn.BCEWithLogitsLoss()

        self.num_nodes = g.ndata["feat"].shape[0]
        if self.num_nodes > 5000:
            self.sample_size = 2000
        else:
            self.sample_size = -1
        self.pseudo_labels = self.get_label().to(device)

    def get_label(self):
        if self.sample_size == -1:
            return torch.cat((torch.ones(self.num_nodes), torch.zeros(self.num_nodes)))
        else:
            return torch.cat((torch.ones(self.sample_size), torch.zeros(self.sample_size)))

    def make_loss(self, x):
        features = self.g.ndata["feat"].to(device)

        self.train()
        if self.sample_size == -1:
            idx = np.arange(self.num_nodes)
        else:
            idx = np.random.default_rng().choice(self.num_nodes, self.sample_size, replace=False)
        c = torch.sigmoid(torch.mean(x[idx], 1))
        h = self.gcn.encoder(self.g, features[np.random.permutation(self.num_nodes), :])[0][-2]
        logits = self.disc(c, x[idx], h[idx])
        loss = self.b_xent(logits, self.pseudo_labels)
        return loss


class Discriminator(nn.Module):
    def __init__(self, hid_dim):
        super(Discriminator, self).__init__()
        self.layer = nn.Bilinear(hid_dim, hid_dim, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def reset_parameters(self):
        for m in self.modules():
            self.weights_init(m)

    def forward(self, c, x, h):
        c = torch.unsqueeze(c, 1).expand_as(x)
        sc_x = torch.squeeze(self.layer(x, c))
        sc_h = torch.squeeze(self.layer(h, c))

        logits = torch.cat((sc_x, sc_h), 0)
        return logits


class PairwiseDistance(nn.Module):
    def __init__(self, g, model, param):
        super(PairwiseDistance, self).__init__()
        self.dataset = param['dataset']
        
        self.disc = nn.Linear(param['hidden_dim'], 4)

        if param['dataset'] != 'ogbn-arxiv':
            self.pseudo_labels = self.get_label(sp.csr_matrix(g.adjacency_matrix(transpose=True).to_dense().cpu().detach().numpy()), 4).to(device)
            self.node_pairs = self.sample(self.pseudo_labels.detach().cpu().numpy() + 1.0)
        else:
            self.pseudo_labels = torch.LongTensor(np.load("../models/saved/node_distance_arxiv_labels.npy")).to(device)
            self.node_pairs = np.load("../models/saved/node_distance_arxiv_pairs.npy")
            print('loading distance matrix...')


    def sample(self, labels, k=4000):

        node_pairs = []

        for i in range(1, int(labels.max()) + 1):
            tmp = np.array(np.where(labels==i)).transpose()
            indices = np.random.choice(np.arange(len(tmp)), k, replace=False)
            node_pairs.append(tmp[indices])

        node_pairs = np.array(node_pairs).reshape(-1, 2).transpose()

        return node_pairs[0], node_pairs[1]


    def make_loss(self, embeddings):

        embeddings0 = embeddings[self.node_pairs[0]]
        embeddings1 = embeddings[self.node_pairs[1]]
        embeddings = self.disc(torch.abs(embeddings0 - embeddings1))

        output = F.log_softmax(embeddings, dim=1)
        if self.dataset != 'ogbn-arxiv':
            loss = F.nll_loss(output, self.pseudo_labels[self.node_pairs])
        else:
            loss = F.nll_loss(output, self.pseudo_labels)

        return loss


    def get_label(self, adj, nclass):
        graph = nx.from_scipy_sparse_matrix(adj)

        if not os.path.exists(f'../models/saved/node_distance_{self.dataset}.npy'):
            path_length = dict(nx.all_pairs_shortest_path_length(graph, cutoff=nclass-1))
            distance = - np.ones((len(graph), len(graph))).astype(int)

            for u, p in path_length.items():
                for v, d in p.items():
                    distance[u][v] = d

            distance[distance==-1] = distance.max() + 1
            distance = np.triu(distance)
            np.save(f'../models/saved/node_distance_{self.dataset}.npy', distance)
        else:
            print('loading distance matrix...')
            distance = np.load(f'../models/saved/node_distance_{self.dataset}.npy')

        return torch.LongTensor(distance) - 1


class PairwiseAttrSim(nn.Module):

    def __init__(self, g, model, param):
        super(PairwiseAttrSim, self).__init__()
        self.dataset = param['dataset']
        self.num_nodes = g.ndata["feat"].shape[0]
        self.seed = param['seed']

        self.disc = nn.Linear(param['hidden_dim'], 2)
        self.build_knn(g.ndata["feat"].cpu(), k=10)

    def build_knn(self, X, k=10):

        if not os.path.exists(f'../models/saved/{self.dataset}_knn_{k}.npz'):
            A_knn = kneighbors_graph(X, k, mode='connectivity', metric='cosine', include_self=True, n_jobs=4)
            print(f'saving saved/{self.dataset}_knn_{k}.npz')
            sp.save_npz(f'../models/saved/{self.dataset}_knn_{k}.npz', A_knn)
        else:
            print(f'loading saved/{self.dataset}_knn_{k}.npz')
            A_knn = sp.load_npz(f'../models/saved/{self.dataset}_knn_{k}.npz')
            
        self.edge_index_knn = torch.LongTensor(A_knn.nonzero())

    def sample(self, n_samples=4000):
        labels = []
        sampled_edges = []

        num_edges = self.edge_index_knn.shape[1]
        idx_selected = np.random.default_rng(self.seed).choice(num_edges, n_samples, replace=False).astype(np.int32)
        labels.append(torch.ones(len(idx_selected), dtype=torch.long))
        sampled_edges.append(self.edge_index_knn[:, idx_selected])

        neg_edges = negative_sampling(edge_index=self.edge_index_knn, num_nodes=self.num_nodes, num_neg_samples=n_samples)
        labels.append(torch.zeros(neg_edges.shape[1], dtype=torch.long))
        sampled_edges.append(neg_edges)
        
        labels = torch.cat(labels).to(device)
        sampled_edges = torch.cat(sampled_edges, axis=1)

        return sampled_edges, labels

    def make_loss(self, embeddings):
        node_pairs, labels = self.sample()
        node_pairs = node_pairs.type(torch.long)
        embeddings0 = embeddings[node_pairs[0]]
        embeddings1 = embeddings[node_pairs[1]]
        embeddings = self.disc(torch.abs(embeddings0 - embeddings1))
        output = F.log_softmax(embeddings, dim=1)
        loss = F.nll_loss(output, labels)
        return loss