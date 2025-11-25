"""
Loss functions for IACD model
"""

import torch
import torch.nn.functional as F


def cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds):
    """BPR (Bayesian Personalized Ranking) Loss"""
    pos_preds = (anc_embeds * pos_embeds).sum(-1)
    neg_preds = (anc_embeds * neg_embeds).sum(-1)
    return torch.sum(F.softplus(neg_preds - pos_preds))


def reg_pick_embeds(embeds_list):
    """L2 regularization for embeddings"""
    reg_loss = 0
    for embeds in embeds_list:
        reg_loss += embeds.square().sum()
    return reg_loss
