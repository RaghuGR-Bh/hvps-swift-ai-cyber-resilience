"""Hybrid Supervised-Unsupervised Pipeline with adversarial robustness test."""
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import yaml

class HybridPipeline:
    def __init__(self, config_path="config.yaml"):
        with open(config_path) as f: self.cfg = yaml.safe_load(f)['models']['hybrid']

    def fuse(self, sup_probs, unsup_scores, y_te, sup_preds=None):
        un = 1-(unsup_scores-unsup_scores.min())/(unsup_scores.max()-unsup_scores.min()+1e-8)
        comb = self.cfg['supervised_weight']*sup_probs + self.cfg['unsupervised_weight']*un
        bt,bf = 0.5,0
        for t in np.arange(0.2,0.8,0.01):
            f=f1_score(y_te,(comb>=t).astype(int))
            if f>bf: bf=f; bt=t
        yh=(comb>=bt).astype(int)
        fp_red=tp_ret=None
        if sup_preds is not None:
            sfp=((sup_preds==1)&(y_te==0)).sum(); hfp=((yh==1)&(y_te==0)).sum()
            fp_red=1-hfp/max(sfp,1)
            stp=((sup_preds==1)&(y_te==1)).sum(); htp=((yh==1)&(y_te==1)).sum()
            tp_ret=htp/max(stp,1)
        return {'model':'hybrid_pipeline','auc_roc':roc_auc_score(y_te,comb),
                'f1':f1_score(y_te,yh),'precision':precision_score(y_te,yh,zero_division=0),
                'recall':recall_score(y_te,yh,zero_division=0),
                'fpr':((yh==1)&(y_te==0)).sum()/(y_te==0).sum(),
                'threshold':bt,'fp_reduction':fp_red,'tp_retention':tp_ret,
                'combined_scores':comb,'predictions':yh}

    def adversarial_test(self, X_te, y_te, sup_model, unsup_model, scale=0.1):
        rng=np.random.RandomState(42)
        sp_c=sup_model.predict_proba(X_te)[:,1]; sd_c=sup_model.predict(X_te)
        sf1_c=f1_score(y_te,sd_c)
        noise=rng.normal(0,scale,X_te.shape); Xp=X_te+noise*np.abs(X_te)
        sp_p=sup_model.predict_proba(Xp)[:,1]; sd_p=sup_model.predict(Xp)
        sf1_p=f1_score(y_te,sd_p); s_deg=(sf1_c-sf1_p)/sf1_c
        # Hybrid with perturbation
        us_c=unsup_model.decision_function(X_te); us_p=unsup_model.decision_function(Xp)
        hc=self.fuse(sp_c,us_c,y_te); hp=self.fuse(sp_p,us_p,y_te)
        h_deg=(hc['f1']-hp['f1'])/hc['f1']
        return {'sup_degradation':s_deg,'hybrid_degradation':h_deg,
                'relative_improvement':(s_deg-h_deg)/s_deg if s_deg>0 else 0}
