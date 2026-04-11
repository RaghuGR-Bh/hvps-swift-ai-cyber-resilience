"""Supervised Models: XGBoost/GradientBoosting, Random Forest, DNN (MLP)."""
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import yaml, warnings
warnings.filterwarnings('ignore')

class SupervisedModels:
    def __init__(self, config_path="config.yaml"):
        with open(config_path) as f: self.cfg = yaml.safe_load(f)['models']
        self.seed = self.cfg['seed']; self.scaler = StandardScaler()
        self.models = {}; self.results = {}

    def _eval(self, model, X_te, y_te, name):
        yp = model.predict_proba(X_te)[:,1]; yd = model.predict(X_te)
        r = {'model':name, 'auc_roc':roc_auc_score(y_te,yp), 'f1':f1_score(y_te,yd),
             'precision':precision_score(y_te,yd,zero_division=0),
             'recall':recall_score(y_te,yd,zero_division=0),
             'fpr':((yd==1)&(y_te==0)).sum()/(y_te==0).sum()}
        self.results[name]=r; return r

    def train_xgboost(self, X_tr, y_tr, X_te, y_te, optimised=False):
        try:
            import xgboost as xgb
            p = self.cfg['xgboost'].copy()
            if optimised: p.update({'n_estimators':800,'max_depth':10,'learning_rate':0.03})
            m = xgb.XGBClassifier(**{k:v for k,v in p.items()},
                random_state=self.seed, use_label_encoder=False, tree_method='hist', n_jobs=-1)
            m.fit(X_tr, y_tr, eval_set=[(X_te,y_te)], verbose=False)
        except ImportError:
            ne = 800 if optimised else 500; lr = 0.03 if optimised else 0.05; md = 10 if optimised else 8
            sw = np.where(y_tr==1,(len(y_tr)-y_tr.sum())/max(y_tr.sum(),1),1.0)
            m = GradientBoostingClassifier(n_estimators=ne,max_depth=md,learning_rate=lr,
                subsample=0.8,random_state=self.seed)
            m.fit(X_tr, y_tr, sample_weight=sw)
        name = 'xgboost_optimised' if optimised else 'xgboost'
        self.models[name] = m
        return self._eval(m, X_te, y_te, name)

    def train_random_forest(self, X_tr, y_tr, X_te, y_te):
        p = self.cfg['random_forest']
        m = RandomForestClassifier(n_estimators=p['n_estimators'],max_depth=p['max_depth'],
            class_weight=p['class_weight'],random_state=self.seed,n_jobs=-1)
        m.fit(X_tr, y_tr); self.models['random_forest'] = m
        return self._eval(m, X_te, y_te, 'random_forest')

    def train_dnn(self, X_tr, y_tr, X_te, y_te):
        Xs = self.scaler.fit_transform(X_tr); Xt = self.scaler.transform(X_te)
        m = MLPClassifier(hidden_layer_sizes=(128,64,32),activation='relu',max_iter=100,
            random_state=self.seed,early_stopping=True)
        m.fit(Xs, y_tr); self.models['dnn'] = m
        yp=m.predict_proba(Xt)[:,1]; yd=m.predict(Xt)
        r = {'model':'dnn','auc_roc':roc_auc_score(y_te,yp),'f1':f1_score(y_te,yd),
             'precision':precision_score(y_te,yd,zero_division=0),
             'recall':recall_score(y_te,yd,zero_division=0),
             'fpr':((yd==1)&(y_te==0)).sum()/(y_te==0).sum()}
        self.results['dnn']=r; return r

    def train_all(self, X_tr, y_tr, X_te, y_te):
        results = []
        for name, fn, kw in [
            ('XGBoost', self.train_xgboost, {'optimised':False}),
            ('XGBoost_opt', self.train_xgboost, {'optimised':True}),
            ('RF', self.train_random_forest, {}),
            ('DNN', self.train_dnn, {}),
        ]:
            print(f"  Training {name}...")
            r = fn(X_tr, y_tr, X_te, y_te, **kw)
            print(f"    AUC: {r.get('auc_roc',0):.3f}  F1: {r['f1']:.3f}")
            results.append(r)
        return results
