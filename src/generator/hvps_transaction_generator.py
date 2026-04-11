
import numpy as np
import pandas as pd
import networkx as nx
import yaml, os

class HVPSTransactionGenerator:
    def __init__(self, config_path="config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg["data"]
        self.rng = np.random.RandomState(self.cfg["seed"])
        self.n_banks = self.cfg["n_banks"]
        self.n_txn = self.cfg["n_transactions"]
        self.fraud_rate = self.cfg["fraud_rate"]
        self.G = nx.barabasi_albert_graph(self.n_banks, m=3, seed=self.cfg["seed"])
        self.centrality = nx.degree_centrality(self.G)
        self.bank_ids = [f"BANK_{i:03d}" for i in range(self.n_banks)]
        raw = self.rng.pareto(1.5, self.n_banks)
        self.bank_weights = raw / raw.sum()

    def _norm(self, w):
        w = np.array(w, dtype=float); return w / w.sum()

    def generate(self, output_path=None):
        n = self.n_txn
        senders = self.rng.choice(self.bank_ids, n, p=self.bank_weights)
        receivers = np.array([self.rng.choice([b for b in self.bank_ids if b!=s],
                    p=self._norm([self.bank_weights[self.bank_ids.index(b)]
                    for b in self.bank_ids if b!=s])) for s in senders])
        values = np.clip(self.rng.lognormal(self.cfg["value_log_mean"],
                 self.cfg["value_log_std"], n), 1e5, 5e8)
        times = np.clip(np.sort(self.rng.normal(self.cfg["peak_hour"], 2.0, n)),
                0, self.cfg["trading_hours"]-0.01)
        ccys = self.rng.choice(["USD","EUR","GBP","JPY","CHF"], n,
               p=[0.45,0.30,0.10,0.08,0.07])
        ptypes = self.rng.choice(["RTGS","COVER","SERIAL","DEBIT","LIQ"], n,
                 p=[0.35,0.25,0.20,0.10,0.10])
        iso = self.rng.beta(8, 2, n)
        df = pd.DataFrame({"transaction_id":[f"TXN_{i:07d}" for i in range(n)],
             "timestamp_hour":times,"sender":senders,"receiver":receivers,
             "amount":values,"currency":ccys,"payment_type":ptypes,
             "iso20022_completeness":iso,"is_fraud":0,"fraud_type":"legitimate"})
        # Inject fraud
        nf = int(n * self.fraud_rate)
        fc = self.cfg["fraud_types"]
        nm = int(nf*fc["message_injection"]); nc = int(nf*fc["credential_transfer"])
        nco = nf - nm - nc
        fi = self.rng.choice(len(df), nf, replace=False)
        # Message injection
        mi = fi[:nm]; df.loc[mi,"is_fraud"]=1; df.loc[mi,"fraud_type"]="message_injection"
        for i in mi:
            sm = df[df.sender==df.loc[i,"sender"]]["amount"].mean()
            df.loc[i,"amount"] = sm * self.rng.uniform(3,15)
        df.loc[mi,"timestamp_hour"] = self.rng.uniform(0,1,nm)
        # Credential
        ci = fi[nm:nm+nc]; df.loc[ci,"is_fraud"]=1; df.loc[ci,"fraud_type"]="credential_transfer"
        for i in ci: df.loc[i,"amount"]=self.rng.choice([1e6,5e6,1e7,5e7])
        df.loc[ci,"iso20022_completeness"]=self.rng.uniform(0.2,0.5,nc)
        # Coordinated
        coi = fi[nm+nc:]; df.loc[coi,"is_fraud"]=1; df.loc[coi,"fraud_type"]="coordinated_multi_entity"
        ring = self.rng.choice(self.bank_ids[-15:],5,replace=False)
        for j,i in enumerate(coi):
            df.loc[i,"sender"]=ring[j%5]; df.loc[i,"receiver"]=ring[(j+1)%5]
            df.loc[i,"amount"]=self.rng.uniform(5e5,5e6)
        # Features
        df["feat_amount"]=df.amount; df["feat_log_amount"]=np.log1p(df.amount)
        df["feat_sender_velocity"]=df.groupby("sender")["transaction_id"].transform("count")
        savg=df.groupby("sender")["amount"].transform("mean"); df["feat_sender_avg_value"]=savg
        sstd=df.groupby("sender")["amount"].transform("std").fillna(1)
        df["feat_value_deviation"]=(df.amount-savg)/sstd
        df["feat_temporal_anomaly"]=np.abs(df.timestamp_hour-self.cfg["peak_hour"])/self.cfg["peak_hour"]
        df["feat_hour_sin"]=np.sin(2*np.pi*df.timestamp_hour/10)
        df["feat_hour_cos"]=np.cos(2*np.pi*df.timestamp_hour/10)
        df["feat_iso_completeness"]=df.iso20022_completeness
        df["feat_value_percentile"]=df.groupby("currency")["amount"].rank(pct=True)
        rc={self.bank_ids[i]:self.centrality[i] for i in range(self.n_banks)}
        df["feat_receiver_centrality"]=df.receiver.map(rc).fillna(0)
        df["feat_sender_centrality"]=df.sender.map(rc).fillna(0)
        df["feat_centrality_ratio"]=df.feat_sender_centrality/(df.feat_receiver_centrality+1e-6)
        df["feat_is_round_amount"]=(df.amount%1e6<1e3).astype(float)
        sccy=df.groupby("sender")["currency"].transform(lambda x:x.mode()[0] if len(x.mode())>0 else "USD")
        df["feat_unusual_currency"]=(df.currency!=sccy).astype(float)
        ptm={"RTGS":0,"COVER":1,"SERIAL":2,"DEBIT":3,"LIQ":4}
        df["feat_payment_type_enc"]=df.payment_type.map(ptm).fillna(0)
        df["feat_sender_seq"]=df.groupby("sender").cumcount()
        df["feat_time_since_last"]=df.groupby("sender")["timestamp_hour"].diff().fillna(0)
        df["feat_rolling_avg"]=df.groupby("sender")["amount"].transform(lambda x:x.rolling(10,min_periods=1).mean())
        df["feat_value_to_rolling"]=df.amount/(df.feat_rolling_avg+1)
        df["feat_coordination_score"]=df.groupby([df.timestamp_hour.round(1),"receiver"])["amount"].transform("count")/10
        pf=df.groupby(["sender","receiver"]).size().reset_index(name="pc")
        st=df.groupby("sender").size().reset_index(name="st")
        pf=pf.merge(st,on="sender"); pf["pr"]=pf.pc/pf.st
        df=df.merge(pf[["sender","receiver","pr"]],on=["sender","receiver"],how="left")
        df["feat_counterparty_deviation"]=1-df.pr.fillna(0); df.drop("pr",axis=1,inplace=True)
        df = df.sample(frac=1, random_state=self.cfg["seed"]).reset_index(drop=True)
        if output_path is None: output_path=self.cfg["output_path"]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        return df

    def get_feature_columns(self):
        return ["feat_amount","feat_log_amount","feat_sender_velocity","feat_sender_avg_value",
                "feat_value_deviation","feat_counterparty_deviation","feat_temporal_anomaly",
                "feat_hour_sin","feat_hour_cos","feat_iso_completeness","feat_value_percentile",
                "feat_receiver_centrality","feat_sender_centrality","feat_centrality_ratio",
                "feat_is_round_amount","feat_unusual_currency","feat_payment_type_enc",
                "feat_sender_seq","feat_time_since_last","feat_rolling_avg",
                "feat_value_to_rolling","feat_coordination_score"]
