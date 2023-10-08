import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops, negative_sampling
from torch_sparse import SparseTensor
from torch.utils.data import DataLoader
from loss import *


def edgeidx2sparse(edge_index, num_nodes):
    return SparseTensor.from_edge_index(
        edge_index, sparse_sizes=(num_nodes, num_nodes)
    ).to(edge_index.device)


def creat_gnn_layer(name, first_channels, second_channels, heads):
    if name == "gcn":
        layer = GCNConv(first_channels, second_channels)
    else:
        raise ValueError(name)
    return layer


def creat_activation_layer(activation):
    if activation is None:
        return nn.Identity()
    if activation == "elu":
        return nn.ELU()
    if activation == "relu":
        return nn.ReLU()
    else:
        raise ValueError("Unknown activation")

class GNNEncoder(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            out_channels,
            num_layers=2,
            dropout=0.5,
            bn=False,
            layer="gcn",
            activation="elu",
            use_node_feats=True,
    ):

        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        bn = nn.BatchNorm1d if bn else nn.Identity
        self.use_node_feats = use_node_feats

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            heads = 1 if i == num_layers - 1 or 'gat' not in layer else 4

            self.convs.append(creat_gnn_layer(layer, first_channels, second_channels, heads))
            self.bns.append(bn(second_channels * heads))

        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x)
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
        x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = self.activation(x)
        return x




class Con_Projector(nn.Module):
    """Simple MLP Edge Decoder"""

    def __init__(
            self, in_channels, hidden_channels, out_channels=1,
            num_layers=2, dropout=0.5, activation='relu'
    ):

        super().__init__()

        self.proj = nn.Linear(in_channels, in_channels)
    def forward(self, x):
        x = self.proj(x)
        return x

class Projector(nn.Module):
    """Simple MLP Decoder"""

    def __init__(
            self, in_channels, hidden_channels, out_channels=1,
            num_layers=2, dropout=0.5, activation='relu'
    ):

        super().__init__()
        self.mlps = nn.ModuleList()

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.mlps.append(nn.Linear(first_channels, second_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)

    def forward(self, x):
        for i, mlp in enumerate(self.mlps[:-1]):
            x = self.dropout(x)
            x = mlp(x)
            x = self.activation(x)
        x = self.dropout(x)   
        x = self.mlps[-1](x)
        x = self.activation(x)  
        return x



def random_negative_sampler(edge_index, num_nodes, num_neg_samples):
    neg_edges = torch.randint(0, num_nodes, size=(2, num_neg_samples)).to(edge_index)
    return neg_edges

class EdgeDecoder(nn.Module):
    """Simple MLP Edge Decoder"""

    def __init__(
            self, in_channels, hidden_channels, out_channels=1,
            num_layers=2, dropout=0.5, activation='relu'
    ):

        super().__init__()
        self.mlps = nn.ModuleList()

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.mlps.append(nn.Linear(first_channels, second_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)

    def forward(self, z_1, z_2, edge, sigmoid=True, reduction=False):
        x = z_1[edge[0]] * z_2[edge[1]]

        if reduction:
            x = x.mean(1)

        for i, mlp in enumerate(self.mlps[:-1]):
            x = self.dropout(x)
            x = mlp(x)
            x = self.activation(x)
        x = self.mlps[-1](x)

        if sigmoid:
            return x.sigmoid()
        else:
            return x

    
class Feature_learner(nn.Module):
    def __init__(self, node, feature):
        super(Feature_learner, self).__init__()

        self.node = node
        self.feature = feature
        self.feature_init = torch.eye(self.node).cuda()
        self.fc = nn.Parameter(torch.randn((self.node, self.feature)))

    def forward(self, x):
        z = torch.mm(self.feature_init, self.fc)
        return z



class Model(nn.Module):
    def __init__(
            self,
            encoder,
            edge_decoder,
            projector,
            con_projector,
            temp,
            pos_weight_tensor, neg_weight_tensor,
            mask=None,
            random_negative_sampling=False,
            loss="ce",
    ):
        super().__init__()
        self.encoder = encoder
        self.edge_decoder = edge_decoder
        self.projector = projector
        self.con_projector = con_projector
        self.mask = mask
        self.temp = temp
        self.pos_weight_tensor = pos_weight_tensor
        self.neg_weight_tensor = neg_weight_tensor

        if loss == "ce":
            self.loss_edgefn = ce_loss
        else:
            raise ValueError(loss)
        self.contrastive_loss = calc_loss
        self.rec_loss = fts_rec_loss

        if random_negative_sampling:
            self.negative_sampler = random_negative_sampler
        else:
            self.negative_sampler = negative_sampling


    def forward(self, data_1, data_2, norm_adj, feature_learner, train_fts_idx, vali_test_fts_idx):

        x_1_, edge_index_1 = data_1.x, data_1.edge_index
        x_learn = feature_learner(x_1_)
        zero_ = torch.zeros_like(x_learn, device=x_learn.device)
        zero = torch.zeros_like(x_learn, device=x_learn.device)
        zero[vali_test_fts_idx] = zero_[vali_test_fts_idx] + x_learn[vali_test_fts_idx]
        x_1__ = x_1_ + zero
        x_1 = torch.mm(norm_adj, x_1__)
        x_2, edge_index_2 = data_2.x, data_2.edge_index

        z_1 = self.encoder(x_1, edge_index_1)
        z_2 = self.encoder(x_2, edge_index_1)
        z = (z_1 + z_2) * 0.5
        out = self.projector(z)
        return out


    def train_one_epoch(
            self, data_1, data_2, norm_adj, feature_learner, train_fts_idx, vali_test_fts_idx, batch_size=2 ** 16):

        x_1_, edge_index_1 = data_1.x, data_1.edge_index
        x_learn = feature_learner(x_1_)
        zero_ = torch.zeros_like(x_learn, device=x_learn.device)
        zero = torch.zeros_like(x_learn, device=x_learn.device)
        zero[vali_test_fts_idx] = zero_[vali_test_fts_idx] + x_learn[vali_test_fts_idx]
        x_1__ = x_1_ + zero
        x_1 = torch.mm(norm_adj, x_1__)
        x_2, edge_index_2 = data_2.x, data_2.edge_index
        remaining_edges, masked_edges = self.mask(edge_index_1)

        aug_edge_index, _ = add_self_loops(edge_index_1)
        neg_edges = self.negative_sampler(
            aug_edge_index,
            num_nodes=data_1.num_nodes,
            num_neg_samples=masked_edges.view(2, -1).size(1),
        ).view_as(masked_edges)

        
        for perm in DataLoader(
                range(masked_edges.size(1)), batch_size=batch_size, shuffle=True
        ):
            z_1 = self.encoder(x_1, remaining_edges)
            z_2 = self.encoder(x_2, remaining_edges)

            batch_masked_edges = masked_edges[:, perm]
            batch_neg_edges = neg_edges[:, perm]

            pos_out_1 = self.edge_decoder(
                z_1, z_2, batch_masked_edges, sigmoid=False
            )
            neg_out_1 = self.edge_decoder(z_1, z_2, batch_neg_edges, sigmoid=False)

            pos_out_2 = self.edge_decoder(
                z_2, z_1, batch_masked_edges, sigmoid=False
            )
            neg_out_2 = self.edge_decoder(z_2, z_1, batch_neg_edges, sigmoid=False)

            loss_edge = (self.loss_edgefn(pos_out_1, neg_out_1) + self.loss_edgefn(pos_out_2, neg_out_2))


            z_1_p = z_1
            z_2_p = z_2
            loss_con = self.contrastive_loss(z_1_p, z_2_p, temperature=self.temp)
            z = (z_1 + z_2) * 0.5


            x_recon = self.projector(z)
            loss_recon = self.rec_loss(x_recon[train_fts_idx], x_1_[train_fts_idx], self.pos_weight_tensor,
                                       self.neg_weight_tensor)    

            loss_total = loss_edge + loss_con + loss_recon         


        return loss_total



