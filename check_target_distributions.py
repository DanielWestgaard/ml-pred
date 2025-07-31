#!/usr/bin/env python3
"""Quick script to check target distributions and understand the discrepancy."""

import pandas as pd
import numpy as np
from utils.logging_utils import get_logger

logger = get_logger()

# Load the data
data_path = "storage/data/capital_com/GBPUSD/processed/1m/2024-01-01_2025-03-01.csv"
data = pd.read_csv(data_path)

print("🔍 CHECKING TARGET DISTRIBUTIONS:")
print(f"Total samples: {len(data)}")

# Check all target columns
target_columns = [col for col in data.columns if col.startswith('target_')]
print(f"\nFound target columns: {target_columns}")

for target in target_columns:
    print(f"\n📊 {target}:")
    value_counts = data[target].value_counts().sort_index()
    print(f"  Distribution: {value_counts.to_dict()}")
    print(f"  Unique values: {data[target].unique()}")
    
    # Check if it's the directional target we care about
    if target == 'target_spread_compression_v3_directional':
        print(f"\n🎯 DIRECTIONAL TARGET ANALYSIS:")
        # If this is truly a 3-class directional target, it should have values [0, 1, 2]
        # where 0=HOLD, 1=BUY, 2=SELL
        unique_vals = sorted(data[target].unique())
        print(f"  Unique values (sorted): {unique_vals}")
        
        if len(unique_vals) == 3 and unique_vals == [0, 1, 2]:
            print("  ✅ This is a proper 3-class directional target")
            class_0_pct = (value_counts[0] / len(data)) * 100
            class_1_pct = (value_counts[1] / len(data)) * 100
            class_2_pct = (value_counts[2] / len(data)) * 100
            print(f"  Class 0 (HOLD): {value_counts[0]} samples ({class_0_pct:.2f}%)")
            print(f"  Class 1 (BUY):  {value_counts[1]} samples ({class_1_pct:.2f}%)")
            print(f"  Class 2 (SELL): {value_counts[2]} samples ({class_2_pct:.2f}%)")
        else:
            print(f"  ❌ Unexpected values for directional target: {unique_vals}")

print("\n🔍 COMPARING WITH DIAGNOSTIC RESULTS:")
print("The diagnostic was run on 'target_spread_compression_v3_long_score' by mistake")
print("But the model was trained on 'target_spread_compression_v3_directional'")
print("This explains the discrepancy!")