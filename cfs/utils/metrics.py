# dtd/utils/metrics.py
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def auroc_ood_high(id_scores, ood_scores) -> float:
    """AUROC where OOD is the positive class (label=1) and higher score means more OOD."""
    id_scores = np.asarray(id_scores).reshape(-1)
    ood_scores = np.asarray(ood_scores).reshape(-1)
    y = np.concatenate([
        np.zeros_like(id_scores, dtype=np.int32),
        np.ones_like(ood_scores, dtype=np.int32),
    ])
    s = np.concatenate([id_scores, ood_scores])
    return float(roc_auc_score(y, s))


def aupr_ood_high(id_scores, ood_scores) -> float:
    """AUPR where OOD is the positive class (label=1) and higher score means more OOD."""
    id_scores = np.asarray(id_scores).reshape(-1)
    ood_scores = np.asarray(ood_scores).reshape(-1)
    y = np.concatenate([
        np.zeros_like(id_scores, dtype=np.int32),
        np.ones_like(ood_scores, dtype=np.int32),
    ])
    s = np.concatenate([id_scores, ood_scores])
    return float(average_precision_score(y, s))


def fpr95_ood_high(id_scores, ood_scores, tpr_target=0.95) -> float:
    """
    FPR@TPR where OOD is the positive class (label=1) and higher score means more OOD.

    Standard OOD setting:
      - positives = OOD
      - negatives = ID
      - threshold chosen so that TPR on OOD reaches `tpr_target`
      - report FPR on ID at that operating point
    """
    id_scores = np.asarray(id_scores).reshape(-1)
    ood_scores = np.asarray(ood_scores).reshape(-1)

    y = np.concatenate([
        np.zeros_like(id_scores, dtype=np.int32),
        np.ones_like(ood_scores, dtype=np.int32),
    ])
    s = np.concatenate([id_scores, ood_scores])

    fpr, tpr, _ = roc_curve(y, s, pos_label=1)

    idx = np.where(tpr >= float(tpr_target))[0]
    if len(idx) == 0:
        return 1.0
    return float(fpr[idx[0]])


def fpr_at_tpr(id_scores, ood_scores, tpr_target=0.95, id_higher=True) -> float:
    """
    Legacy helper in the ID-positive convention.
    Positive = ID. If id_higher=False, scores are negated first.
    """
    id_scores = np.asarray(id_scores).ravel()
    ood_scores = np.asarray(ood_scores).ravel()
    if not id_higher:
        id_scores = -id_scores
        ood_scores = -ood_scores

    thr = np.quantile(id_scores, 1.0 - float(tpr_target))
    fpr = (ood_scores >= thr).mean()
    return float(fpr)
