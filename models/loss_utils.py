import torch as t
import torch.nn.functional as F

"""
Basic Loss Functions
"""

def cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds):
    """BPR (Bayesian Personalized Ranking) Loss"""
    pos_preds = (anc_embeds * pos_embeds).sum(-1)
    neg_preds = (anc_embeds * neg_embeds).sum(-1)
    return t.sum(F.softplus(neg_preds - pos_preds))


def reg_pick_embeds(embeds_list):
    """L2 regularization for embeddings"""
    reg_loss = 0
    for embeds in embeds_list:
        reg_loss += embeds.square().sum()
    return reg_loss


def reg_params(model):
    """L2 regularization for model parameters"""
    reg_loss = 0
    for W in model.parameters():
        reg_loss += W.norm(2).square()
    return reg_loss


"""
Self-supervised Learning Loss Functions
"""

def cal_infonce_loss(embeds1, embeds2, all_embeds2, temp=1.0):
    """InfoNCE Loss"""
    normed_embeds1 = embeds1 / t.sqrt(1e-8 + embeds1.square().sum(-1, keepdim=True))
    normed_embeds2 = embeds2 / t.sqrt(1e-8 + embeds2.square().sum(-1, keepdim=True))
    normed_all_embeds2 = all_embeds2 / t.sqrt(1e-8 + all_embeds2.square().sum(-1, keepdim=True))
    nume_term = -(normed_embeds1 * normed_embeds2 / temp).sum(-1)
    deno_term = t.log(t.sum(t.exp(normed_embeds1 @ normed_all_embeds2.T / temp), dim=-1))
    cl_loss = (nume_term + deno_term).sum()
    return cl_loss


def cal_infonce_loss_spec_nodes(embeds1, embeds2, nodes, temp):
    """InfoNCE Loss (specify nodes for contrastive learning)"""
    embeds1 = F.normalize(embeds1 + 1e-8, p=2)
    embeds2 = F.normalize(embeds2 + 1e-8, p=2)
    pckEmbeds1 = embeds1[nodes]
    pckEmbeds2 = embeds2[nodes]
    nume = t.exp(t.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temp)
    deno = t.exp(pckEmbeds1 @ embeds2.T / temp).sum(-1) + 1e-8
    return -t.log(nume / deno).mean()


def alignment(x, y, alpha=2):
    """Alignment Loss (proposed by DirectAU)"""
    x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniformity(x):
    """Uniformity Loss (proposed by DirectAU)"""
    x = F.normalize(x, dim=-1)
    return t.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()
