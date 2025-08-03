#!/usr/bin/env python3
"""
Create Final ML Features - Guaranteed to work
"""

import pandas as pd
import numpy as np

def main():
    print("🔍 Creating Final ML Dataset...")
    
    # Load transaction data
    df = pd.read_csv("data/demo/transaction_network.csv")
    print(f"📊 Loaded {len(df):,} transactions")
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sample 20,000 transactions for ML features
    ml_df = df.sample(n=20000, random_state=42).copy()
    
    # Add time-based features
    ml_df['hour'] = ml_df['timestamp'].dt.hour
    ml_df['day_of_week'] = ml_df['timestamp'].dt.dayofweek
    ml_df['is_weekend'] = (ml_df['timestamp'].dt.dayofweek >= 5).astype(int)
    ml_df['is_night'] = ((ml_df['timestamp'].dt.hour >= 22) | (ml_df['timestamp'].dt.hour <= 6)).astype(int)
    
    # Add product risk
    product_risk = {'PIX': 0.7, 'TED': 0.5, 'DOC': 0.3, 'DEBIT': 0.2}
    ml_df['product_risk'] = ml_df['product_type'].map(product_risk).fillna(0.5)
    
    # Add amount categories
    ml_df['amount_log'] = np.log1p(ml_df['amount'])
    ml_df['is_high_amount'] = (ml_df['amount'] > ml_df['amount'].quantile(0.9)).astype(int)
    
    # Select final features
    feature_cols = [
        'transaction_id', 'customer_id', 'amount', 'amount_log', 'hour', 
        'day_of_week', 'is_weekend', 'is_night', 'product_risk', 
        'is_high_amount', 'location_risk', 'is_fraud'
    ]
    
    final_df = ml_df[feature_cols].copy()
    
    # Save
    final_df.to_csv("data/demo/ml_features.csv", index=False)
    
    print(f"✅ Created ML dataset with {len(final_df):,} samples")
    print(f"📊 Features: {len(feature_cols)} columns")
    print(f"⚠️ Fraud rate: {final_df['is_fraud'].mean():.2%}")
    print("💾 Saved to: data/demo/ml_features.csv")
    
    # Show sample
    print("\n📋 Sample data:")
    print(final_df.head())
    
    return final_df

if __name__ == "__main__":
    main()