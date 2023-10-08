import random
import scipy.sparse as sp
import torch
import torch.nn as nn
import sys
import pickle as pkl
import torch.nn.functional as F
import pickle
import os
import yaml
from sklearn.svm import SVC
import numpy as np
from sklearn.utils import shuffle
import networkx as nx
from scipy.linalg import fractional_matrix_power, inv
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_npz_to_sparse_graph(file_name):
    with np.load(file_name) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']), shape=loader['adj_shape'])

        if 'attr_data' in loader:
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']), shape=loader['attr_shape'])
        elif 'attr_matrix' in loader:
            attr_matrix = loader['attr_matrix']
        else:
            attr_matrix = None

        if 'labels_data' in loader:
            labels = sp.csr_matrix((loader['labels_data'], loader['labels_indices'], loader['labels_indptr']), shape=loader['labels_shape'])
        elif 'labels' in loader:
            labels = loader['labels']
        else:
            labels = None

        node_names = loader.get('node_names')
        attr_names = loader.get('attr_names')
        class_names = loader.get('class_names')
        metadata = loader.get('metadata')

    return SparseGraph(adj_matrix, attr_matrix, labels, node_names, attr_names, class_names, metadata)


class SparseGraph:
    def __init__(self, adj_matrix, attr_matrix=None, labels=None,
                 node_names=None, attr_names=None, class_names=None, metadata=None):

        if sp.isspmatrix(adj_matrix):
            adj_matrix = adj_matrix.tocsr().astype(np.float32)
        else:
            raise ValueError("Adjacency matrix must be in sparse format (got {0} instead)".format(type(adj_matrix)))

        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("Dimensions of the adjacency matrix don't agree")

        if attr_matrix is not None:
            if sp.isspmatrix(attr_matrix):
                attr_matrix = attr_matrix.tocsr().astype(np.float32)
            elif isinstance(attr_matrix, np.ndarray):
                attr_matrix = attr_matrix.astype(np.float32)
            else:
                raise ValueError("Attribute matrix must be a sp.spmatrix or a np.ndarray (got {0} instead)".format(type(attr_matrix)))

            if attr_matrix.shape[0] != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency and attribute matrices don't agree")

        if labels is not None:
            if labels.shape[0] != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency matrix and the label vector don't agree")

        if node_names is not None:
            if len(node_names) != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency matrix and the node names don't agree")

        if attr_names is not None:
            if len(attr_names) != attr_matrix.shape[1]:
                raise ValueError("Dimensions of the attribute matrix and the attribute names don't agree")

        self.adj_matrix = adj_matrix
        self.attr_matrix = attr_matrix
        self.labels = labels
        self.node_names = node_names
        self.attr_names = attr_names
        self.class_names = class_names
        self.metadata = metadata

    def num_nodes(self):
        return self.adj_matrix.shape[0]

    def num_edges(self):
        if self.is_directed():
            return int(self.adj_matrix.nnz)
        else:
            return int(self.adj_matrix.nnz / 2)

    def get_neighbors(self, idx):
        return self.adj_matrix[idx].indices

    def is_directed(self):
        return (self.adj_matrix != self.adj_matrix.T).sum() != 0

    def to_undirected(self):
        if self.is_weighted():
            raise ValueError("Convert to unweighted graph first.")
        else:
            self.adj_matrix = self.adj_matrix + self.adj_matrix.T
            self.adj_matrix[self.adj_matrix != 0] = 1
        return self

    def is_weighted(self):
        return np.any(np.unique(self.adj_matrix[self.adj_matrix != 0].A1) != 1)

    def to_unweighted(self):
        self.adj_matrix.data = np.ones_like(self.adj_matrix.data)
        return self

    def unpack(self):
        return self.adj_matrix, self.attr_matrix, self.labels


def compute_ppr(a, alpha=0.2, self_loop=True):
    if self_loop:
        a = a + np.eye(a.shape[0])  # A^ = A + I_n
    d = np.diag(np.sum(a, 1))  # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)  # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)  # A~ = D^(-1/2) x A^ x D^(-1/2)
    return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))  # a(I_n-(1-a)A~)^-1


def load_data(args):
    print('loading dataset: {}'.format(args.dataset))
    if args.dataset in ['cora', 'citeseer']:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("./data/{}/ind.{}.{}".format(args.dataset, args.dataset, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))
        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file("./data/{}/ind.{}.test.index".format(args.dataset, args.dataset))
        test_idx_range = np.sort(test_idx_reorder)
        if args.dataset == 'citeseer':
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        datadir = os.path.join(f'./data/{args.dataset}/diff.npy')
        if not os.path.exists(datadir):
            adj_numpy = nx.to_numpy_array(nx.from_dict_of_lists(graph))
            diff = compute_ppr(adj_numpy, 0.2)
            np.save(f'./data/{args.dataset}/diff.npy', diff)
        else:
            diff = np.load(f'./data/{args.dataset}/diff.npy')
        diff = torch.FloatTensor(diff)

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        labels = np.argmax(labels, 1)
        labels = torch.from_numpy(labels).long()
        if not args.generative_flag:
            features = normalize_features(features)
        adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
        indices_norm = torch.from_numpy(np.stack([adj_norm.tocoo().row, adj_norm.tocoo().col], axis=0).astype(float)).long()
        values_norm = torch.from_numpy(adj_norm.tocoo().data.astype(float)).float()
        adj_norm = torch.sparse.FloatTensor(indices_norm, values_norm, torch.Size(adj_norm.shape))
        indices = torch.from_numpy(np.stack([adj.tocoo().row, adj.tocoo().col], axis=0).astype(float)).long()
        values = torch.from_numpy(adj.tocoo().data.astype(float)).float()
        adj = torch.sparse.FloatTensor(indices, values, torch.Size(adj.shape))
        features = torch.from_numpy(np.array(features.todense())).float()
    elif args.dataset in ['amac']:
        data = load_npz_to_sparse_graph(os.path.join(os.getcwd(), 'data', 'amac', 'amazon_electronics_computers.npz'))
        features = data.attr_matrix.todense()
        if not args.generative_flag:
            features = normalize_features(features)
        features = torch.from_numpy(features).float()
        adj = data.adj_matrix
        adj = adj + adj.T
        adj.data = np.ones_like(adj.data)
        adj = adj.tocoo()

        datadir = os.path.join(f'./data/{args.dataset}/diff.npy')
        if not os.path.exists(datadir):
            adj_numpy = np.array(adj.toarray())
            diff = compute_ppr(adj_numpy, 0.2)
            np.save(f'./data/{args.dataset}/diff.npy', diff)
        else:
            diff = np.load(f'./data/{args.dataset}/diff.npy')
        diff = torch.FloatTensor(diff)

        adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
        indices_norm = torch.from_numpy(np.stack([adj_norm.tocoo().row, adj_norm.tocoo().col], axis=0).astype(float)).long()
        values_norm = torch.from_numpy(adj_norm.tocoo().data.astype(float)).float()
        adj_norm = torch.sparse.FloatTensor(indices_norm, values_norm, torch.Size(adj_norm.shape))
        indices = torch.from_numpy(np.stack([adj.tocoo().row, adj.tocoo().col], axis=0).astype(float)).long()
        values = torch.from_numpy(adj.data).float()
        adj = torch.sparse.FloatTensor(indices, values, torch.Size(adj.shape))
        labels = torch.from_numpy(data.labels).long()
    elif args.dataset in ['amap']:
        data = load_npz_to_sparse_graph(os.path.join(os.getcwd(), 'data', 'amap', 'amazon_electronics_photo.npz'))
        features = data.attr_matrix.todense()
        if not args.generative_flag:
            features = normalize_features(features)
        features = torch.from_numpy(features).float()
        adj = data.adj_matrix
        adj = adj + adj.T
        adj.data = np.ones_like(adj.data)
        adj = adj.tocoo()
        #
        datadir = os.path.join(f'./data/{args.dataset}/diff.npy')
        if not os.path.exists(datadir):
            adj_numpy = np.array(adj.toarray())
            diff = compute_ppr(adj_numpy, 0.2)
            np.save(f'./data/{args.dataset}/diff.npy', diff)
        else:
            diff = np.load(f'./data/{args.dataset}/diff.npy')
        diff = torch.FloatTensor(diff)

        adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
        indices_norm = torch.from_numpy(np.stack([adj_norm.tocoo().row, adj_norm.tocoo().col], axis=0).astype(float)).long()
        values_norm = torch.from_numpy(adj_norm.tocoo().data.astype(float)).float()
        adj_norm = torch.sparse.FloatTensor(indices_norm, values_norm, torch.Size(adj_norm.shape))
        indices = torch.from_numpy(np.stack([adj.tocoo().row, adj.tocoo().col], axis=0).astype(float)).long()
        values = torch.from_numpy(adj.data).float()
        adj = torch.sparse.FloatTensor(indices, values, torch.Size(adj.shape))
        labels = torch.from_numpy(data.labels).long()
        
    else:
        print('Cannot process this dataset!')
        raise Exception

    return adj, diff, adj_norm, features, labels, indices


def load_generated_features(path):
    fts = pkl.load(open(path, 'rb'))
    norm_fts = normalize_features(fts)
    norm_fts = torch.from_numpy(norm_fts).float()
    return norm_fts


def normalize_adj(mx):
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def cal_accuracy(train_fts, train_lbls, test_fts, test_lbls):
    clf = SVC(gamma='auto')
    clf.fit(train_fts, train_lbls)

    preds_lbls = clf.predict(test_fts)
    acc = accuracy(preds_lbls, test_lbls)
    return acc


def RECALL_NDCG(estimated_fts, true_fts, topN=10):
    preds = np.argsort(-estimated_fts, axis=1)
    preds = preds[:, :topN]

    gt = [np.where(true_fts[i, :] != 0)[0] for i in range(true_fts.shape[0])]
    recall_list = []
    ndcg_list = []
    for i in range(preds.shape[0]):
        if len(gt[i]) != 0:
            if np.sum(estimated_fts[i, :]) != 0:
                recall = len(set(preds[i, :]) & set(gt[i])) * 1.0 / len(set(gt[i]))
                recall_list.append(recall)

                intersec = np.array(list(set(preds[i, :]) & set(gt[i])))
                if len(intersec) > 0:
                    dcg = [np.where(preds[i, :] == ele)[0] for ele in intersec]
                    dcg = np.sum([1.0 / (np.log2(x + 1 + 1)) for x in dcg])
                    idcg = np.sum([1.0 / (np.log2(x + 1 + 1)) for x in range(len(gt[i]))])
                    ndcg = dcg * 1.0 / idcg
                else:
                    ndcg = 0.0
                ndcg_list.append(ndcg)
            else:
                temp_preds = shuffle(np.arange(estimated_fts.shape[1]))[:topN]

                recall = len(set(temp_preds) & set(gt[i])) * 1.0 / len(set(gt[i]))
                recall_list.append(recall)

                intersec = np.array(list(set(temp_preds) & set(gt[i])))
                if len(intersec) > 0:
                    dcg = [np.where(temp_preds == ele)[0] for ele in intersec]
                    dcg = np.sum([1.0 / (np.log2(x + 1 + 1)) for x in dcg])
                    idcg = np.sum([1.0 / (np.log2(x + 1 + 1)) for x in range(len(gt[i]))])
                    ndcg = dcg * 1.0 / idcg
                else:
                    ndcg = 0.0
                ndcg_list.append(ndcg)

    avg_recall = np.mean(recall_list)
    avg_ndcg = np.mean(ndcg_list)

    return avg_recall, avg_ndcg


class MLP(nn.Module):
    def __init__(self, fts_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(fts_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_fts):
        h1 = F.relu(self.fc1(input_fts))
        h2 = self.fc2(h1)
        return F.log_softmax(h2, dim=1)


def class_eva(train_fts, train_lbls, test_fts, test_lbls):
    test_featured_idx = np.where(test_fts.sum(1) != 0)[0]
    test_non_featured_idx = np.where(test_fts.sum(1) == 0)[0]

    featured_test_fts = test_fts[test_featured_idx]
    featured_test_lbls = test_lbls[test_featured_idx]
    non_featured_test_lbls = test_lbls[test_non_featured_idx]

    fts_dim = train_fts.shape[1]
    hid_dim = 64
    n_class = int(max(max(train_lbls), max(test_lbls)) + 1)
    is_cuda = torch.cuda.is_available()

    model = MLP(fts_dim, hid_dim, n_class)
    if is_cuda:
        model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    featured_test_lbls_arr = featured_test_lbls.copy()

    train_fts = torch.from_numpy(train_fts).float()
    train_lbls = torch.from_numpy(train_lbls).long()
    featured_test_fts = torch.from_numpy(featured_test_fts).float()
    featured_test_lbls = torch.from_numpy(featured_test_lbls).long()
    if is_cuda:
        train_fts = train_fts.cuda()
        train_lbls = train_lbls.cuda()
        featured_test_fts = featured_test_fts.cuda()
        featured_test_lbls = featured_test_lbls.cuda()

    acc_list = []
    for i in range(1000):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_fts)

        loss = F.nll_loss(outputs, train_lbls)
        loss.backward()
        optimizer.step()

        model.eval()
        featured_test_outputs = model(featured_test_fts)
        test_loss = F.nll_loss(featured_test_outputs, featured_test_lbls)
        if is_cuda:
            featured_test_outputs = featured_test_outputs.data.cpu().numpy()
        else:
            featured_test_outputs = featured_test_outputs.data.numpy()
        featured_preds = np.argmax(featured_test_outputs, axis=1)

        random_preds = np.random.choice(n_class, len(test_non_featured_idx))

        preds = np.concatenate((featured_preds, random_preds))
        lbls = np.concatenate((featured_test_lbls_arr, non_featured_test_lbls))

        acc = np.sum(preds == lbls) * 1.0 / len(lbls)
        acc_list.append(acc)
        # print('Epoch: {}, train loss: {:.4f}, test loss: {:.4f}, test acc: {:.4f}'.format(i, loss.item(), test_loss.item(), acc))

    # print('Best epoch:{}, best acc: {:.4f}'.format(np.argmax(acc_list), np.max(acc_list)))
    return np.max(acc_list)




def data_split(args, adj):
    shuffled_nodes = shuffle(np.arange(adj.shape[0]), random_state=72)
    train_fts_idx = torch.from_numpy(shuffled_nodes[:int(args.train_fts_ratio * adj.shape[0])]).long()
    vali_fts_idx = torch.from_numpy(
        shuffled_nodes[
        int(args.train_fts_ratio * adj.shape[0]):int((args.train_fts_ratio + 0.1) * adj.shape[0])]).long()
    test_fts_idx = torch.from_numpy(shuffled_nodes[int((args.train_fts_ratio + 0.1) * adj.shape[0]):]).long()
    vali_test_fts_idx = torch.from_numpy(shuffled_nodes[int(args.train_fts_ratio * adj.shape[0]):]).long()
    print("Dataset loading done!")
    return train_fts_idx, vali_fts_idx, test_fts_idx, vali_test_fts_idx


def loss_weight(args, true_features, train_fts_idx):
    if args.dataset in ['cora', 'citeseer', 'amac', 'amap', 'steam']:
        fts_loss_func = fts_loss_discrete
        pos_weight = torch.sum(true_features[train_fts_idx] == 0.0).item() / (
            torch.sum(true_features[train_fts_idx] != 0.0).item())
    else:
        fts_loss_func = None
        pos_weight = None
        print("Error!")
    if args.cuda:
        pos_weight_tensor = torch.from_numpy(np.array([pos_weight])).float().cuda()
        neg_weight_tensor = torch.from_numpy(np.array([1.0])).float().cuda()
    else:
        pos_weight_tensor = torch.from_numpy(np.array([pos_weight])).float()
        neg_weight_tensor = torch.from_numpy(np.array([1.0])).float()
    return fts_loss_func, pos_weight_tensor, neg_weight_tensor


def input_matrix(args, adj, norm_adj, true_features):
    indices = torch.from_numpy(np.stack([np.arange(adj.shape[0]), np.arange(adj.shape[0])], axis=0)).long()
    values = torch.from_numpy(np.ones(indices.shape[1])).float()
    diag_fts = torch.sparse.FloatTensor(indices, values, torch.Size([adj.shape[0], adj.shape[0]]))
    if args.cuda:
        A = norm_adj.cuda()
        D = diag_fts.to_dense().cuda()
        true_features = true_features.cuda()
    else:
        A = norm_adj
        D = diag_fts.to_dense()
        true_features = true_features
    A_temp = A
    return A, D, true_features, A_temp




def graph_loss_func(graph_recon=None, pos_indices=None, neg_indices=None, pos_values=None, neg_values=None):
    BCE = torch.nn.BCEWithLogitsLoss(reduction='none')
    loss_indices = torch.cat([pos_indices, neg_indices], dim=0)
    preds_logits = graph_recon[loss_indices[:, 0], loss_indices[:, 1]]
    labels = torch.cat([pos_values, neg_values])
    loss_bce = torch.mean(BCE(preds_logits, labels))
    return loss_bce


def fts_loss_discrete(recon_x=None, x=None, p_weight=None, n_weight=None):
    BCE = torch.nn.BCEWithLogitsLoss(reduction='none')
    output_fts_reshape = torch.reshape(recon_x, shape=[-1])
    out_fts_lbls_reshape = torch.reshape(x, shape=[-1])
    weight_mask = torch.where(out_fts_lbls_reshape != 0.0, p_weight, n_weight)
    loss_bce = torch.mean(BCE(output_fts_reshape, out_fts_lbls_reshape) * weight_mask)
    return loss_bce




def save_generative_fts(args, gene_X, T, train_fts_idx, vali_fts_idx, test_fts_idx):
    if args.dataset in ['cora', 'citeseer', 'amap', 'amac']:
        output_fts = gene_X
    else:
        output_fts = None
        print("Error!")
    if args.cuda:
        train_fts = T[train_fts_idx].data.cpu().numpy()
        vali_fts = T[vali_fts_idx].data.cpu().numpy()
        train_fts_idx_arr = train_fts_idx.cpu().numpy()
        vali_fts_idx_arr = vali_fts_idx.cpu().numpy()
        test_fts_idx_arr = test_fts_idx.cpu().numpy()
    else:
        train_fts = T[train_fts_idx].data.numpy()
        vali_fts = T[vali_fts_idx].data.numpy()
        train_fts_idx_arr = train_fts_idx.numpy()
        vali_fts_idx_arr = vali_fts_idx.numpy()
        test_fts_idx_arr = test_fts_idx.numpy()
    save_fts = np.zeros(shape=T.shape)
    save_fts[train_fts_idx_arr] = train_fts
    save_fts[vali_fts_idx_arr] = vali_fts
    save_fts[test_fts_idx_arr] = output_fts
    pickle.dump(save_fts, open(os.path.join(os.getcwd(), 'features', 'final_gene_fts_train_ratio_{}_{}.pkl'.format(args.dataset, args.train_fts_ratio)), 'wb'))



def test_model(args, model, norm_adj, feature_learn, T, data_1, data_2, train_id, vali_id, vali_test_id, test_id):
    print('Loading well-trained model'.format(args.epoch))

    model.load_state_dict(
        torch.load(os.path.join(os.getcwd(), 'model', 'final_model_{}_{}.pkl'.format(args.dataset, args.train_fts_ratio))))
    feature_learn.load_state_dict(
        torch.load(os.path.join(os.getcwd(), 'model', 'ft_learn_model_{}_{}.pkl'.format(args.dataset, args.train_fts_ratio))))
    
    model.eval()
    feature_learn.eval()

    with torch.no_grad():
        X_hat = model(data_1, data_2, norm_adj, feature_learn, train_id, vali_test_id)
    gene_fts = X_hat[test_id]

    reture_recall = 0.0
    reture_ndcg = 0.0
    print('Profiling performance on {}:'.format(args.dataset))
    if args.cuda:
        gene_fts = gene_fts.data.cpu().numpy()
        gt_fts = T[test_id].cpu().numpy()
    else:
        gene_fts = gene_fts.data.numpy()
        gt_fts = T[test_id].numpy()
    for topK in args.topK_list:
        avg_recall, avg_ndcg = RECALL_NDCG(gene_fts, gt_fts, topN=topK)
        print('topK: {}, recall: {}, ndcg: {}'.format(topK, avg_recall, avg_ndcg))
        if topK == 50:
            reture_recall = avg_recall
            reture_ndcg = avg_ndcg
    save_generative_fts(args, gene_fts, T, train_id, vali_id, test_id)
    if args.cuda:
        T = T.cpu().data.numpy()
    else:
        T = T.data.numpy()
    
    return reture_recall, reture_ndcg
    

def set_random_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)
    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args

def mask_edge(edge_index, p=0.7):
    if p < 0. or p > 1.:
        raise ValueError(f'Mask probability has to be between 0 and 1 '
                         f'(got {p}')
    e_ids = torch.arange(edge_index.size(1), dtype=torch.long, device=edge_index.device)
    mask = torch.full_like(e_ids, p, dtype=torch.float32)
    # print(mask)
    # print(mask.size())
    mask_ = torch.bernoulli(mask).to(torch.bool)
    # print(mask.size())

    return edge_index[:, ~mask_], edge_index[:, mask_]

class MaskEdge(nn.Module):
    def __init__(self, p: float=0.7, undirected: bool=True):
        super().__init__()
        self.p = p
        self.undirected = undirected

    def forward(self, edge_index):
        remaining_edges, masked_edges = mask_edge(edge_index, p=self.p)
        if self.undirected:
            remaining_edges = to_undirected(remaining_edges)
        return remaining_edges, masked_edges

    def extra_repr(self):
        return f"p={self.p}, undirected={self.undirected}"




import os
import pickle
from sklearn.utils import shuffle
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
import torch
import torch.optim as optim
import torch.nn as nn
# from utils import normalize_adj
from sklearn.model_selection import KFold

def normalize_adj(mx):
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)



def test_X(gene_fts, labels_of_gene):

    print('begining test_X......')

    gene_data = np.concatenate((gene_fts, np.reshape(labels_of_gene, newshape=[-1, 1])), axis=1)
    final_list = []
    for i in range(10):
        gene_data = shuffle(gene_data, random_state=72)
        KF = KFold(n_splits=5)
        split_data = KF.split(gene_data)
        acc_list = []
        for train_idx, test_idx in split_data:
            train_data = gene_data[train_idx]
            train_featured_idx = np.where(train_data.sum(1) != 0)[0]
            train_data = train_data[train_featured_idx]
            test_data = gene_data[test_idx]
            acc = class_eva(train_fts=train_data[:, :-1], train_lbls=train_data[:, -1],
                            test_fts=test_data[:, :-1], test_lbls=test_data[:, -1])
            acc_list.append(acc)
        avg_acc = np.mean(acc_list)
        final_list.append(avg_acc)
    print('classification performance: {}'.format(np.mean(final_list)))

class GCNLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout):
        super(GCNLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(in_features, out_features).type(
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),
            requires_grad=True)

    def forward(self, input, sp_adj, is_sp_fts=False):
        if is_sp_fts:
            h = torch.spmm(input, self.W)
        else:
            h = torch.mm(input, self.W)
        h_prime = torch.spmm(sp_adj, h)
        return F.elu(h_prime)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GCN_eva(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, input_fts_sparse=True):
        """Dense version of GAT."""
        super(GCN_eva, self).__init__()
        self.dropout = dropout
        self.GCNlayer1 = GCNLayer(nfeat, nhid, dropout=dropout)
        self.GCNlayer2 = GCNLayer(nhid, nhid, dropout=dropout)
        self.input_fts_sparse = input_fts_sparse

        self.fc1 = nn.Linear(nhid, nclass)

    def forward(self, x, sp_adj):
        h1 = self.GCNlayer1(x, sp_adj, is_sp_fts=self.input_fts_sparse)
        h1 = F.dropout(h1, self.dropout, training=self.training)
        self.z = self.GCNlayer2(h1, sp_adj, is_sp_fts=False)

        h3 = F.log_softmax(self.fc1(self.z), dim=1)

        return h3

def test_AX(gene_data, labels_of_gene, adj):

    train_fts_ratio = 0.4 * 1.0
    print('begining test_AX......')

    is_cuda = torch.cuda.is_available()

    n_nodes = adj.shape[0]
    indices = np.where(adj != 0)
    rows = indices[0]
    cols = indices[1]
    adj = sp.coo_matrix((np.ones(shape=len(rows)), (rows, cols)), shape=[n_nodes, n_nodes])
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    indices = torch.LongTensor(np.int64(np.stack([adj.tocoo().row, adj.tocoo().col], axis=0)))
    values = torch.FloatTensor(adj.tocoo().data)
    adj = torch.sparse.FloatTensor(indices, values, torch.Size(adj.shape))
    labels_of_gene = torch.LongTensor(labels_of_gene)
    n_class = max(labels_of_gene).item() + 1
    features = torch.FloatTensor(gene_data)

    final_list = []
    for i in range(10):
        node_Idx = shuffle(np.arange(labels_of_gene.shape[0]), random_state=72)
        KF = KFold(n_splits=5)
        split_data = KF.split(node_Idx)
        acc_list = []
        for train_idx, test_idx in split_data:
            train_idx = torch.LongTensor(train_idx)
            test_idx = torch.LongTensor(test_idx)
            train_fts = features[train_idx]
            test_fts = features[test_idx]
            featured_train_idx = train_idx[(train_fts.sum(1) != 0).nonzero().reshape([-1])]
            featured_test_idx = test_idx[(test_fts.sum(1) != 0).nonzero().reshape([-1])]
            non_featured_test_idx = test_idx[(test_fts.sum(1) == 0).nonzero().reshape([-1])]
            featured_train_lbls = labels_of_gene[featured_train_idx]
            featured_test_lbls = labels_of_gene[featured_test_idx]
            non_featured_test_lbls = labels_of_gene[non_featured_test_idx]
            featured_test_lbls_arr = featured_test_lbls.numpy()
            non_featured_test_lbls_arr = non_featured_test_lbls.numpy()
            model = GCN_eva(nfeat=features.shape[1], nhid=64, nclass=n_class, dropout=0.1, input_fts_sparse=False)
            if is_cuda:
                model.cuda()
                adj = adj.cuda()
                features = features.cuda()
                featured_train_lbls = featured_train_lbls.cuda()
                featured_test_lbls = featured_test_lbls.cuda()
                featured_train_idx = featured_train_idx.cuda()
                featured_test_idx = featured_test_idx.cuda()
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            best_acc = 0
            for epoch in range(1000):
                model.train()
                optimizer.zero_grad()
                output = model(features, adj)
                loss_train = F.nll_loss(output[featured_train_idx], featured_train_lbls)
                loss_train.backward()
                optimizer.step()
                model.eval()
                val_loss = F.nll_loss(output[featured_test_idx], featured_test_lbls)
                if is_cuda:
                    featured_preds = np.argmax(output[featured_test_idx].data.cpu().numpy(), axis=1)
                else:
                    featured_preds = np.argmax(output[featured_test_idx].data.numpy(), axis=1)
                random_preds = np.random.choice(np.arange(n_class), len(non_featured_test_idx))
                preds = np.concatenate((featured_preds, random_preds))
                lbls = np.concatenate((featured_test_lbls_arr, non_featured_test_lbls_arr))
                acc = np.sum(preds == lbls) * 1.0 / len(preds)
                if acc > best_acc:
                    best_acc = acc


            acc_list.append(best_acc)
        avg_acc = np.mean(acc_list)
        final_list.append(avg_acc)
    print('GCN(A+X),  avg accuracy: {}'.format(np.mean(final_list)))