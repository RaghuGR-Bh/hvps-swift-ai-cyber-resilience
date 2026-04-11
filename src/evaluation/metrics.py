
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

def comprehensive_metrics(y_true, y_pred, y_prob=None, name="model"):
    cm=confusion_matrix(y_true,y_pred); tn,fp,fn,tp=cm.ravel()
    r={"model":name,"precision":precision_score(y_true,y_pred,zero_division=0),
       "recall":recall_score(y_true,y_pred,zero_division=0),
       "f1":f1_score(y_true,y_pred),"fpr":fp/(fp+tn) if (fp+tn)>0 else 0}
    if y_prob is not None: r["auc_roc"]=roc_auc_score(y_true,y_prob)
    return r
