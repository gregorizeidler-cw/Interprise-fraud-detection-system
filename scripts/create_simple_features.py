#!/usr/bin/env python3
"""
Create Simple ML Features from Demo Data - No errors version
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_simple_features():
    """Create simple ML features"""
    
    print("🔍 Creating Simple ML Features...")
    
    # Load data
    transactions_df = pd.read_csv("data/demo/transaction_network.csv")
    customers_df = pd.read_csv("data/demo/customer_profiles.csv")
    
    print(f"📊 Loaded {len(transactions_df):,} transactions")
    print(f"👥 Loaded {len(customers_df):,} customers")
    
    # Convert timestamp
    transactions_df['timestamp'] = pd.to_datetime(transactions_df['timestamp'])
    
    # Sample for faster processing
    sample_df = transactions_df.sample(n=15000, random_state=42).copy()
    
    # Create basic features directly from transactions
    sample_df['hour_of_day'] = sample_df['timestamp'].dt.hour
    sample_df['day_of_week'] = sample_df['timestamp'].dt.dayofweek
    sample_df['is_weekend'] = (sample_df['timestamp'].dt.dayofweek >= 5).astype(int)
    sample_df['is_night'] = ((sample_df['timestamp'].dt.hour >= 22) | 
                            (sample_df['timestamp'].dt.hour <= 6)).astype(int)
    
    # Product risk scores
    product_risk_map = {'PIX': 0.7, 'TED': 0.5, 'DOC': 0.3, 'DEBIT': 0.2}
    sample_df['product_risk'] = sample_df['product_type'].map(product_risk_map).fillna(0.5)
    
    # Add customer info
    sample_df = sample_df.merge(customers_df[['customer_id', 'age', 'income', 'credit_score', 
                                            'account_age_days', 'is_pep', 'high_risk_location', 'risk_score']], 
                               on='customer_id', how='left')
    
    # Rename columns for clarity
    feature_columns = [
        'transaction_id', 'customer_id', 'amount', 'hour_of_day', 'day_of_week', 
        'is_weekend', 'is_night', 'product_risk', 'age', 'income', 'credit_score',
        'account_age_days', 'is_pep', 'high_risk_location', 'risk_score', 'location_risk', 'is_fraud'
    ]
    
    # Select and clean features
    final_df = sample_df[feature_columns].copy()
    final_df = final_df.dropna()
    
    # Save
    final_df.to_csv("data/demo/ml_features.csv", index=False)
    
    # Create summary
    summary = {
        'total_samples': len(final_df),
        'fraud_rate': float(final_df['is_fraud'].mean()),
        'avg_amount': float(final_df['amount'].mean()),
        'features': list(final_df.columns)
    }
    
    import json
    with open("data/demo/features_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Simple ML Features created!")
    print(f"📊 {len(final_df):,} samples with {len(final_df.columns)} features")
    print(f"⚠️ Fraud rate: {final_df['is_fraud'].mean():.2%}")
    print(f"💾 Saved to: data/demo/ml_features.csv")
    
    return final_df

if __name__ == "__main__":
    create_simple_features()