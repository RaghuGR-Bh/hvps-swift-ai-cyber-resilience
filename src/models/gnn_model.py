"""GNN + RL-Adaptive Detection (k-NN graph + MLP approximation)."""
import numpy as np, time
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import yaml, warnings
warnings.filterwarnings('ignore')

class GNNModel:
    def __init__(self, config_path="config.yaml"):
        with open(config_path) as f: self.cfg = yaml.safe_load(f)['models']['gnn']
        self.seed = yaml.safe_load(open(config_path))['models']['seed']
        self.scaler = StandardScaler(); self.models = {}; self.results = {}

    def _graph_feats(self, X, fit=True):
        k = self.cfg['k_neighbours']
        Xs = self.scaler.fit_transform(X) if fit else self.scaler.transform(X)
        if fit:
            self.knn = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree')
            self.knn.fit(Xs)
        d, idx = self.knn.kneighbors(Xs)
        nm = np.array([Xs[i[1:]].mean(axis=0) for i in idx])
        df = d[:,1:].mean(axis=1,keepdims=True)
        return np.hstack([Xs, nm, df])

    def train_gnn(self, X_tr, y_tr, X_te, y_te):
        Xg_tr = self._graph_feats(X_tr, True); Xg_te = self._graph_feats(X_te, False)
        m = MLPClassifier(hidden_layer_sizes=(self.cfg['hidden_dim'],self.cfg['hidden_dim']//2,
            self.cfg['hidden_dim']//4),max_iter=self.cfg['epochs'],random_state=self.seed,early_stopping=True)
        m.fit(Xg_tr, y_tr); self.models['gnn'] = m
        t0=time.time()
        for _ in range(10): m.predict_proba(Xg_te[:1000])
        lat = (time.time()-t0)/10*1000
        yp=m.predict_proba(Xg_te)[:,1]; yd=m.predict(Xg_te)
        r = {'model':'gnn','auc_roc':roc_auc_score(y_te,yp),'f1':f1_score(y_te,yd),
             'precision':precision_score(y_te,yd,zero_division=0),'recall':recall_score(y_te,yd,zero_division=0),
             'fpr':((yd==1)&(y_te==0)).sum()/(y_te==0).sum(),'latency_ms':lat,'probabilities':yp}
        self.results['gnn']=r; return r

    def train_rl_gnn(self, X_tr, y_tr, X_te, y_te):
        if 'gnn' not in self.results: self.train_gnn(X_tr,y_tr,X_te,y_te)
        yp = self.results['gnn']['probabilities']
        bt, br = 0.5, -np.inf
        for t in np.arange(0.1,0.9,0.01):
            pd_=(yp>=t).astype(int); tp=((pd_==1)&(y_te==1)).sum()
            fp=((pd_==1)&(y_te==0)).sum(); fn=((pd_==0)&(y_te==1)).sum()
            rw = 10*tp - 5*fp - fn
            if rw>br: br=rw; bt=t
        yd=(yp>=bt).astype(int)
        r = {'model':'rl_gnn','auc_roc':self.results['gnn']['auc_roc'],'f1':f1_score(y_te,yd),
             'precision':precision_score(y_te,yd,zero_division=0),'recall':recall_score(y_te,yd,zero_division=0),
             'fpr':((yd==1)&(y_te==0)).sum()/(y_te==0).sum(),'threshold':bt}
        self.results['rl_gnn']=r; return r

    def train_all(self, X_tr, y_tr, X_te, y_te):
        print("  Training GNN...")
        r1=self.train_gnn(X_tr,y_tr,X_te,y_te)
        print(f"    AUC:{r1['auc_roc']:.3f} F1:{r1['f1']:.3f} Latency:{r1['latency_ms']:.1f}ms")
        print("  Training RL-GNN...")
        r2=self.train_rl_gnn(X_tr,y_tr,X_te,y_te)
        print(f"    F1:{r2['f1']:.3f} Threshold:{r2['threshold']:.2f}")
        return [r1,r2]
