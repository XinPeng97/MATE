import torch
import torch.nn.functional as F



def fts_rec_loss(recon_x=None, x=None, p_weight=None, n_weight=None):
    BCE = torch.nn.BCEWithLogitsLoss(reduction='none')
    output_fts_reshape = torch.reshape(recon_x, shape=[-1])
    out_fts_lbls_reshape = torch.reshape(x, shape=[-1])
    weight_mask = torch.where(out_fts_lbls_reshape != 0.0, p_weight, n_weight)
    loss_bce = torch.mean(BCE(output_fts_reshape, out_fts_lbls_reshape) * weight_mask)
    return loss_bce


def ce_loss(pos_out, neg_out):
    pos_loss = F.binary_cross_entropy(pos_out.sigmoid(), torch.ones_like(pos_out))
    neg_loss = F.binary_cross_entropy(neg_out.sigmoid(), torch.zeros_like(neg_out))
    return pos_loss + neg_loss

def calc_loss(x, x_aug, temperature=2.0, sym=True):
    batch_size = x.shape[0]
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / (torch.einsum('i,j->ij', x_abs, x_aug_abs) + 1e-8)

    sim_matrix = torch.exp(sim_matrix / temperature)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]

    if sym:
        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss_0 = - torch.log(loss_0).mean()
        loss_1 = - torch.log(loss_1).mean()
        loss = (loss_0 + loss_1) / 2.0
    else:
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
    return loss


