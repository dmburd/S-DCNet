import sys
from sys import exit as e

import torch
import torch.nn as nn
import torch.nn.functional as F
import q


def counter_loss(gt_cls_labels_list, pred_cls_logits_list):
    cross_entropy_loss_terms = []
    gt_pred_pairs = zip(gt_cls_labels_list, pred_cls_logits_list)
    
    for gt_cls_i_labels, pred_cls_i_logits in gt_pred_pairs:
        t = nn.CrossEntropyLoss()(
            pred_cls_i_logits,
            gt_cls_i_labels.to(device=pred_cls_i_logits.device, dtype=torch.long))
        cross_entropy_loss_terms.append(t)

    return cross_entropy_loss_terms
    

def merging_loss(gt_div2, pred_div2):
    Lm = nn.L1Loss()(pred_div2, gt_div2.to(pred_div2.device))
    return Lm


def upsampling_loss(counts_gt, U1, U2):
    count0_gt, count1_gt, count2_gt = counts_gt
    krn = torch.ones((count0_gt.shape[0], 1, 2, 2))
    U1_gt = count1_gt / F.conv_transpose2d(count0_gt, krn, stride=2)
    U2_gt = count2_gt / F.conv_transpose2d(count1_gt, krn, stride=2)
    # U1_gt and U2_gt may contain nan values because of 0/0 divisions;
    # in such cases the nans should be replaced by 0.25
    U1_gt.masked_fill_(torch.isnan(U1_gt), 0.25)
    U2_gt.masked_fill_(torch.isnan(U2_gt), 0.25)
    Lup1 = nn.L1Loss()(U1, U1_gt.to(U1.device))
    Lup2 = nn.L1Loss()(U2, U2_gt.to(U2.device))
    return Lup1 + Lup2


def division_loss(counts_gt, W1, W2, Cmax):
    indic = [(1 * (c > Cmax)).to(W1.device) for c in counts_gt]
    Ldiv1 = -indic[0] * torch.log(F.max_pool2d(W1, kernel_size=2, stride=2))
    Ldiv2 = -indic[1] * torch.log(F.max_pool2d(W2, kernel_size=2, stride=2))
    Ldiv_sum = Ldiv1.sum() + Ldiv2.sum()
    return Ldiv_sum
    
    
def total_loss(losses_list):
    total = 0
    for component in losses_list:
        if isinstance(component, list):
            total += sum(component)
        else:
            total += component
    # print('all loss components:')
    # for a in losses_list:
    #    print(a)
    # print()
    return total
    
    