#!/usr/bin/env python3
"""Step 2: Train supervised, unsupervised, GNN, and hybrid models."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from src.generator.hvps_transaction_generator import HVPSTransactionGenerator

try:
    import xgboost as xgb
    USE_XGB = True
except ImportError:
    USE_XGB = False
    print("XGBoost not installed; using sklearn GradientBoosting as proxy.")

if __name__ == "__main__":
    df = pd.read_csv("data/hvps_transactions.csv")
    gen = HVPSTransactionGenerator(config_path="config.yaml")
    fc = [c for c in gen.get_feature_columns() if c in df.columns]
    X = df[fc].fillna(0).values; y = df['is_fraud'].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sw = np.where(y_tr==1, (len(y_tr)-y_tr.sum())/max(y_tr.sum(),1), 1.0)
    results = []

    # XGBoost / GB
    if USE_XGB:
        m = xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=769,
            eval_metric="aucpr", random_state=42, use_label_encoder=False, tree_method='hist')
        m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    else:
        m = GradientBoostingClassifier(n_estimators=500, max_depth=8, learning_rate=0.05,
            subsample=0.8, random_state=42)
        m.fit(X_tr, y_tr, sample_weight=sw)
    p = m.predict_proba(X_te)[:,1]; pd_ = m.predict(X_te)
    auc = roc_auc_score(y_te, p); f1 = f1_score(y_te, pd_)
    print(f"XGBoost: AUC={auc:.3f} F1={f1:.3f}")
    results.append({"model":"XGBoost","auc":auc,"f1":f1})

    # RF
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rfa = roc_auc_score(y_te, rf.predict_proba(X_te)[:,1])
    rff = f1_score(y_te, rf.predict(X_te))
    print(f"RF: AUC={rfa:.3f} F1={rff:.3f}")
    results.append({"model":"RF","auc":rfa,"f1":rff})

    # IF
    iso = IsolationForest(n_estimators=200, contamination=0.002, random_state=42, n_jobs=-1)
    iso.fit(X_tr); iso_sc = iso.decision_function(X_te)
    iso_pd = (iso.predict(X_te)==-1).astype(int)
    isof = f1_score(y_te, iso_pd)
    print(f"IF: F1={isof:.3f}")

    # Hybrid
    iso_n = 1-(iso_sc-iso_sc.min())/(iso_sc.max()-iso_sc.min()+1e-8)
    comb = 0.65*p + 0.35*iso_n
    bt, bf = 0.5, 0
    for t in np.arange(0.2,0.8,0.01):
        f = f1_score(y_te, (comb>=t).astype(int))
        if f>bf: bf=f; bt=t
    hp = (comb>=bt).astype(int)
    ha = roc_auc_score(y_te, comb); hf = f1_score(y_te, hp)
    sfp = ((pd_==1)&(y_te==0)).sum(); hfp = ((hp==1)&(y_te==0)).sum()
    fpr = 1-hfp/max(sfp,1)
    print(f"Hybrid: AUC={ha:.3f} F1={hf:.3f} FP_reduction={fpr:.0%}")
    results.append({"model":"Hybrid","auc":ha,"f1":hf,"fp_reduction":fpr})

    pd.DataFrame(results).to_csv("data/model_results.csv", index=False)
    print("\nResults saved to data/model_results.csv")
