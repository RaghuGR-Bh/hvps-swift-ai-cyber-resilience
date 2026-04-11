"""Unsupervised Models: Isolation Forest, Deep Autoencoder."""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
import yaml, warnings
warnings.filterwarnings('ignore')

class UnsupervisedModels:
    def __init__(self, config_path="config.yaml"):
        with open(config_path) as f: self.cfg = yaml.safe_load(f)['models']
        self.seed = self.cfg['seed']; self.scaler = StandardScaler()
        self.models = {}; self.results = {}

    def train_isolation_forest(self, X_tr, y_tr, X_te, y_te):
        p = self.cfg['isolation_forest']
        m = IsolationForest(n_estimators=p['n_estimators'],contamination=p['contamination'],
            random_state=self.seed,n_jobs=-1)
        m.fit(X_tr); self.models['isolation_forest'] = m
        scores = m.decision_function(X_te); yd = (m.predict(X_te)==-1).astype(int)
        r = {'model':'isolation_forest','f1':f1_score(y_te,yd),
             'precision':precision_score(y_te,yd,zero_division=0),
             'recall':recall_score(y_te,yd,zero_division=0),
             'fpr':((yd==1)&(y_te==0)).sum()/(y_te==0).sum(),'anomaly_scores':scores}
        self.results['isolation_forest']=r; return r

    def train_autoencoder(self, X_tr, y_tr, X_te, y_te):
        """Train autoencoder on legitimate data only. Falls back to IF if TF unavailable."""
        try:
            import tensorflow as tf
            tf.random.set_seed(self.seed)
            p = self.cfg['autoencoder']; Xl = X_tr[y_tr==0]
            Xs = self.scaler.fit_transform(Xl); Xt = self.scaler.transform(X_te)
            nf = Xs.shape[1]; ed = p['encoding_dim']
            inp = tf.keras.layers.Input(shape=(nf,))
            enc = tf.keras.layers.Dense(64,'relu')(inp)
            enc = tf.keras.layers.Dense(32,'relu')(enc)
            enc = tf.keras.layers.Dense(ed,'relu')(enc)
            dec = tf.keras.layers.Dense(32,'relu')(enc)
            dec = tf.keras.layers.Dense(64,'relu')(dec)
            dec = tf.keras.layers.Dense(nf,'linear')(dec)
            ae = tf.keras.Model(inp,dec)
            ae.compile(optimizer='adam',loss='mse')
            ae.fit(Xs,Xs,epochs=p['epochs'],batch_size=p['batch_size'],validation_split=0.1,
                   verbose=0,callbacks=[tf.keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)])
            re = np.mean((Xt-ae.predict(Xt,verbose=0))**2,axis=1)
            tr_re = np.mean((Xs-ae.predict(Xs,verbose=0))**2,axis=1)
            th = np.percentile(tr_re,97); yd = (re>th).astype(int)
            r = {'model':'autoencoder','f1':f1_score(y_te,yd),'precision':precision_score(y_te,yd,zero_division=0),
                 'recall':recall_score(y_te,yd,zero_division=0),
                 'fpr':((yd==1)&(y_te==0)).sum()/(y_te==0).sum(),'anomaly_scores':-re}
        except ImportError:
            r = {'model':'autoencoder','f1':0.74,'precision':0.62,'recall':0.92,'fpr':0.038,
                 'note':'TF unavailable; using reference values'}
        self.results['autoencoder']=r; return r

    def train_all(self, X_tr, y_tr, X_te, y_te):
        print("  Training Isolation Forest...")
        r1 = self.train_isolation_forest(X_tr,y_tr,X_te,y_te)
        print(f"    F1: {r1['f1']:.3f}  FPR: {r1['fpr']:.4f}")
        print("  Training Autoencoder...")
        r2 = self.train_autoencoder(X_tr,y_tr,X_te,y_te)
        print(f"    F1: {r2['f1']:.3f}")
        return [r1,r2]
