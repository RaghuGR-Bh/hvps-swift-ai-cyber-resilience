#!/usr/bin/env python3
"""Step 1: Generate ~500K synthetic HVPS transactions with 0.13% fraud rate."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.generator.hvps_transaction_generator import HVPSTransactionGenerator

if __name__ == "__main__":
    gen = HVPSTransactionGenerator(config_path="config.yaml")
    df = gen.generate(output_path="data/hvps_transactions.csv")
    nf = df['is_fraud'].sum()
    print(f"\nVerification: {len(df):,} txns, {nf} fraud ({nf/len(df):.4%}), "
          f"{len(gen.get_feature_columns())} features")
