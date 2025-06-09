import torch
from sklearn.metrics import f1_score
from utils import mat2bp
from utils import bp2matrix

def contact_f1(gt_contact, pred_contact, reduce=True, method="triangular"):
    """Compute F1 from base pairs. Input goes to sigmoid and then thresholded"""
    # gb, predはcontact

    f1 = f1_triangular(gt_contact, pred_contact)
    
    return f1


def f1_triangular(gt, pred):
    """Compute F1 from the upper triangular connection matrix"""
    # get upper triangular matrix without diagonal
    ind = torch.triu_indices(gt.shape[0], gt.shape[1], offset=1)

    gt = gt[ind[0], ind[1]].numpy().ravel()
    pred = pred[ind[0], ind[1]].numpy().ravel()

    return f1_score(gt, pred, zero_division=0)


def f1_strict(ref_bp, pre_bp):
    """F1 score strict, same as triangular but less efficient"""
    # corner case when there are no positives
    if len(ref_bp) == 0 and len(pre_bp) == 0:
        return 1.0, 1.0, 1.0

    tp1 = 0
    for rbp in ref_bp:
        if rbp in pre_bp:
            tp1 = tp1 + 1
    tp2 = 0
    for pbp in pre_bp:
        if pbp in ref_bp:
            tp2 = tp2 + 1

    fn = len(ref_bp) - tp1
    fp = len(pre_bp) - tp1

    tpr = pre = f1 = 0.0
    if tp1 + fn > 0:
        tpr = tp1 / float(tp1 + fn)  # sensitivity (=recall =power)
    if tp1 + fp > 0:
        pre = tp2 / float(tp1 + fp)  # precision (=ppv)
    if tpr + pre > 0:
        f1 = 2 * pre * tpr / (pre + tpr)  # F1 score

    return tpr, pre, f1


def f1_shift(ref_bp, pre_bp):
    """F1 score with tolerance of 1 position"""
    # corner case when there are no positives
    if len(ref_bp) == 0 and len(pre_bp) == 0:
        return 1.0, 1.0, 1.0

    tp1 = 0
    for rbp in ref_bp:
        if (
            rbp in pre_bp
            or [rbp[0], rbp[1] - 1] in pre_bp
            or [rbp[0], rbp[1] + 1] in pre_bp
            or [rbp[0] + 1, rbp[1]] in pre_bp
            or [rbp[0] - 1, rbp[1]] in pre_bp
        ):
            tp1 = tp1 + 1
    tp2 = 0
    for pbp in pre_bp:
        if (
            pbp in ref_bp
            or [pbp[0], pbp[1] - 1] in ref_bp
            or [pbp[0], pbp[1] + 1] in ref_bp
            or [pbp[0] + 1, pbp[1]] in ref_bp
            or [pbp[0] - 1, pbp[1]] in ref_bp
        ):
            tp2 = tp2 + 1

    fn = len(ref_bp) - tp1
    fp = len(pre_bp) - tp1

    tpr = pre = f1 = 0.0
    if tp1 + fn > 0:
        tpr = tp1 / float(tp1 + fn)  # sensitivity (=recall =power)
    if tp1 + fp > 0:
        pre = tp2 / float(tp1 + fp)  # precision (=ppv)
    if tpr + pre > 0:
        f1 = 2 * pre * tpr / (pre + tpr)  # F1 score

    return tpr, pre, f1
