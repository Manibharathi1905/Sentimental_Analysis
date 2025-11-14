import numpy as np
from sklearn.metrics import f1_score, classification_report

def threshold_per_class(y_true, y_pred_probs):
    # choose 0.5 defaults; better: tune on validation to maximize per-class F1
    thresh = np.full(y_true.shape[1], 0.5)
    preds = (y_pred_probs >= thresh).astype(int)
    macro = f1_score(y_true, preds, average="macro", zero_division=0)
    micro = f1_score(y_true, preds, average="micro", zero_division=0)
    return {"macro_f1": macro, "micro_f1": micro}
