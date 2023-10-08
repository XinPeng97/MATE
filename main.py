import argparse
from torch import optim
from MATE import *
from utils import *
import warnings
import random
from tqdm import tqdm
from torch_geometric.data import Data

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--method_name', type=str, default='Model')
parser.add_argument('--topK_list', type=list, default=[10, 20, 50])
parser.add_argument('--seed', type=int, default=72)
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.001)  
parser.add_argument('--p', type=float, default=0.7)
parser.add_argument('--weight_decay', type=float, default=5e-5)  
parser.add_argument('--train_fts_ratio', type=float, default=0.4)
parser.add_argument('--generative_flag', type=bool, default=True)
parser.add_argument('--cuda', action='store_true',
                    default=torch.cuda.is_available())

parser.add_argument("--layer", nargs="?", default="gcn",
                    help="GNN layer, (default: gcn)")
parser.add_argument("--encoder_activation", nargs="?", default="elu",
                    help="Activation function for GNN encoder, (default: elu)")
parser.add_argument('--encoder_channels', type=int, default=128,
                    help='Channels of GNN encoder layers. (default: 128)')
parser.add_argument('--hidden_channels', type=int, default=64,
                    help='Channels of hidden representation. (default: 64)')
parser.add_argument('--decoder_channels', type=int, default=32,
                    help='Channels of decoder layers. (default: 32)')
parser.add_argument('--encoder_layers', type=int, default=2,
                    help='Number of layers for encoder. (default: 2)')
parser.add_argument('--eproj_layer', type=int, default=2,
                    help='Number of layers for edge_projector. (default: 2)')
parser.add_argument('--decoder_layers', type=int, default=2,
                    help='Number of layers for decoders. (default: 2)')
parser.add_argument('--encoder_dropout', type=float, default=0.8,
                    help='Dropout probability of encoder. (default: 0.8)')
parser.add_argument('--eproj_dropout', type=float, default=0.2,
                    help='Dropout probability of edge_projector. (default: 0.2)')
parser.add_argument('--decoder_dropout', type=float, default=0.2,
                    help='Dropout probability of decoder. (default: 0.2)')
parser.add_argument('--bn', type=bool, default=False)
parser.add_argument('--device', type=int, default=3)
parser.add_argument('--alpha', type=int, default=1)
parser.add_argument('--beta', type=int, default=1)
parser.add_argument('--temp', type=float, default=0.2)


def main(args):
    set_random_seed(72)
    adj, diff, norm_adj, true_features, node_labels, indices = load_data(args)

    Adj, Diag, Ture_feature, A_temp = input_matrix(
        args, adj, norm_adj, true_features)
    train_id, vali_id, test_id, vali_test_id = data_split(args, adj)


    x_view1_ = true_features
    x_view1_[vali_test_id] = 0.0
    data_1 = Data(x=x_view1_, y=node_labels, edge_index=indices)


    x_view_ = true_features
    x_view_[vali_test_id] = 0.0
    x_view_ = x_view_.cuda()
    diff = diff.cuda()
    x_view2 = torch.mm(diff, x_view_).cpu()

    data_2 = Data(x=x_view2, y=node_labels, edge_index=indices)
    fts_loss_func, pos_weight_tensor, neg_weight_tensor = loss_weight(
        args, true_features, train_id)

    set_random_seed(args.seed)
    mask = MaskEdge(p=args.p)
    feature_learn = Feature_learner(
        true_features.size()[0], true_features.size()[1])
    
    encoder = GNNEncoder(data_1.num_features, args.encoder_channels, args.hidden_channels,
                         num_layers=args.encoder_layers, dropout=args.encoder_dropout,
                         bn=args.bn, layer=args.layer, activation=args.encoder_activation)

    edge_decoder = EdgeDecoder(args.hidden_channels, args.decoder_channels,
                               num_layers=args.eproj_layer, dropout=args.eproj_dropout)

    projector = Projector(args.hidden_channels, args.encoder_channels, out_channels=data_1.num_features,
                          num_layers=args.decoder_layers, dropout=args.decoder_dropout)

    con_projector = Con_Projector(args.hidden_channels, args.encoder_channels, out_channels=data_1.num_features,
                          num_layers=args.decoder_layers, dropout=args.decoder_dropout)

    model = Model(encoder, edge_decoder, projector, con_projector, args.temp, pos_weight_tensor, neg_weight_tensor, mask)

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    optimizer_learner = torch.optim.Adam(
        feature_learn.parameters(), lr=1e-3, weight_decay=args.weight_decay)

    def scheduler(epoch): return (
        1 + np.cos((epoch) * np.pi / args.epoch)) * 0.5
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=scheduler)

    if args.cuda:
        data_1 = data_1.cuda()
        data_2 = data_2.cuda()
        model = model.cuda()
        feature_learn = feature_learn.cuda()
        norm_adj = norm_adj.cuda()

    eva_values_list = []
    best = 0.0
    print('---------------------start trainning------------------------')
    for epoch in tqdm(range(1, 1 + args.epoch)):
        model.train()
        feature_learn.train()

        loss = model.train_one_epoch(data_1, data_2, norm_adj, feature_learn,
                                     train_id, vali_test_id)
        
        optimizer.zero_grad()
        optimizer_learner.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_learner.step()
        scheduler.step()
        if epoch % 20 == 0:
            model.eval()
            feature_learn.eval()
            with torch.no_grad():
                X_hat = model(data_1, data_2, norm_adj, 
                              feature_learn, train_id, vali_test_id)
            gene_fts = X_hat[vali_id].cpu().numpy()
            gt_fts = Ture_feature[vali_id].cpu().numpy()
            avg_recall, avg_ndcg = RECALL_NDCG(
                gene_fts, gt_fts, topN=args.topK_list[2])
            eva_values_list.append(avg_recall)
            if eva_values_list[-1] > best:
                torch.save(model.state_dict(),
                           os.path.join(os.getcwd(), 'model',
                                        'final_model_{}_{}.pkl'.format(args.dataset, args.train_fts_ratio)))                
                torch.save(feature_learn.state_dict(), os.path.join(os.getcwd(), 'model',
                        'ft_learn_model_{}_{}.pkl'.format(args.dataset, args.train_fts_ratio)))
                best = eva_values_list[-1]


    model.load_state_dict(
        torch.load(os.path.join(os.getcwd(), 'model', 'final_model_{}_{}.pkl'.format(args.dataset, args.train_fts_ratio))))
    feature_learn.load_state_dict(
        torch.load(os.path.join(os.getcwd(), 'model', 'ft_learn_model_{}_{}.pkl'.format(args.dataset, args.train_fts_ratio))))    
    
    model.eval()
    feature_learn.eval()
    _,_ = test_model(args, model, norm_adj, feature_learn, Ture_feature,
                                data_1, data_2, train_id, vali_id, vali_test_id, test_id)
    with torch.no_grad():
        x_hat = model(data_1, data_2, norm_adj, feature_learn, train_id, vali_test_id)
        gene_data = x_hat[test_id]
        labels_of_gene = node_labels[test_id]
    adj = adj.to_dense()
    test_X(gene_data.cpu().numpy(), labels_of_gene.cpu().numpy())
    test_AX(gene_data.cpu().numpy(), labels_of_gene.cpu().numpy(), adj[test_id, :][:, test_id].cpu().numpy())
    
if __name__ == "__main__":
    args = parser.parse_args()
    args = load_best_configs(args, "configs.yml")
    print(args)
    torch.cuda.set_device(f'cuda:{args.device}')
    main(args)

