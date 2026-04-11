#!/usr/bin/env python3
"""Step 4: SHAP / feature importance analysis."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from src.generator.hvps_transaction_generator import HVPSTransactionGenerator

if __name__ == "__main__":
    df = pd.read_csv("data/hvps_transactions.csv")
    gen = HVPSTransactionGenerator(config_path="config.yaml")
    fc = [c for c in gen.get_feature_columns() if c in df.columns]
    X = df[fc].fillna(0).values; y = df['is_fraud'].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sw = np.where(y_tr==1, (len(y_tr)-y_tr.sum())/max(y_tr.sum(),1), 1.0)

    gb = GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=42)
    gb.fit(X_tr, y_tr, sample_weight=sw)

    try:
        import shap
        explainer = shap.TreeExplainer(gb)
        sv = explainer.shap_values(X_te[:3000])
        imp = np.abs(sv).mean(axis=0)
        idf = pd.DataFrame({"feature":fc,"shap":imp}).sort_values("shap",ascending=False)
        print("Top features (SHAP):")
        for _,r in idf.head(8).iterrows(): print(f"  {r.feature}: {r.shap:.4f}")
        plt.figure(figsize=(10,6))
        shap.summary_plot(sv, X_te[:3000], feature_names=fc, show=False, max_display=12)
        plt.tight_layout(); plt.savefig("data/shap_summary.png", dpi=150); plt.close()
        idf.to_csv("data/shap_feature_importance.csv", index=False)
        print("SHAP plots saved to data/")
    except ImportError:
        imp = gb.feature_importances_
        idf = pd.DataFrame({"feature":fc,"importance":imp}).sort_values("importance",ascending=False)
        print("Top features (feature_importances_):")
        for _,r in idf.head(8).iterrows(): print(f"  {r.feature}: {r.importance:.4f}")
        plt.figure(figsize=(10,6))
        t15 = idf.head(15)
        plt.barh(range(len(t15)), t15.importance.values, color='#2E5E8E')
        plt.yticks(range(len(t15)), t15.feature.values)
        plt.xlabel('Feature Importance'); plt.title('Top 15 Features')
        plt.gca().invert_yaxis(); plt.tight_layout()
        plt.savefig("data/feature_importance.png", dpi=150); plt.close()
        idf.to_csv("data/shap_feature_importance.csv", index=False)
        print("Feature importance saved to data/")
