#!/usr/bin/env python3
"""Step 3: Run cyber stress-test scenarios S1-S4 and Basel capital analysis."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.stress_test.cyber_stress_engine import CyberStressEngine

if __name__ == "__main__":
    e = CyberStressEngine(config_path="config.yaml")
    st = e.run_all()
    st.to_csv("data/stress_test_results.csv", index=False)
    print("\nScenario Results:")
    for _,r in st.iterrows():
        print(f"  {r.scenario}: {r.reduction_pct:.1f}% reduction | Recovery: {r.ai_recovery_h:.1f}h")
    print(f"\n  Range: {st.reduction_pct.min():.0f}%-{st.reduction_pct.max():.0f}%")

    print("\nBasel Capital Impact (30% loss reduction):")
    b = e.basel_capital_impact(0.30)
    for k,v in b.items():
        print(f"  {k}: {v}")
