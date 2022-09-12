import numpy as np
import torch

from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.metrics import hamming_loss, zero_one_loss, coverage_error
from sklearn.metrics import average_precision_score, label_ranking_loss


def multi_label_metrics(probs, targs):
    probs = probs.cpu().data.numpy()
    targs = targs.cpu().data.numpy()
    preds = (probs >= 0.5).astype(int)
    n_tasks = targs.shape[1]
    label = np.arange(2)

    ham_l = hamming_loss(targs, preds)
    zero_one = zero_one_loss(targs, preds)
    sub_acc = 1. - zero_one
    cover = coverage_error(targs, probs)
    rank_l = label_ranking_loss(targs, probs)
    avgprec = average_precision_score(targs, probs)

    results = []
    overall = []

    for i in range(n_tasks):
        targ = targs[:, i]
        prob = probs[:, i]
        pred = preds[:, i]

        cm = confusion_matrix(targ, pred, labels=label).ravel()
        overall.append(cm)
        tn, fp, fn, tp = cm
        try:
            reca = tp / (tp + fn) if tp+fn != 0. else 0.
            prec = tp / (tp + fp) if tp+fp != 0. else 0.
            spec = tn / (tn + fp) if tn+fp != 0. else 0.
            f1 = 2*reca*prec/(reca+prec) if reca+prec != 0. else 0.
        except:
            import pdb; pdb.set_trace()

        acc = (tn + tp) / (tn + tp + fn + fp)
        auc = roc_auc_score(y_true=targ, y_score=prob)
        
        current = [acc, reca, prec, spec, f1, auc]
        results.append(current)
    results = np.asarray(results)
    aucs = results[..., -1]
    mean = np.mean(results, axis=0)
    pc_recall = mean[1]
    pc_precision = mean[2]

    mean_acc = mean[0]
    macro_f1 = mean[4]
    mean_auc = mean[5]

    return np.asarray([sub_acc, ham_l, rank_l, avgprec, 
            pc_recall, pc_precision, macro_f1, mean_auc]), aucs






