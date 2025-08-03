#!/usr/bin/env python3
"""
Create ML Features from Demo Data
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_ml_features():
    """Create ML-ready feature set from demo data"""
    
    print("🔍 Creating ML Features from Demo Data...")
    
    # Load data
    logger.info("Loading demo data...")
    customers_df = pd.read_csv("data/demo/customer_profiles.csv")
    transactions_df = pd.read_csv("data/demo/transaction_network.csv")
    
    # Convert timestamp
    transactions_df['timestamp'] = pd.to_datetime(transactions_df['timestamp'])
    
    # Sample for faster processing
    sample_size = 25000
    sampled_txns = transactions_df.sample(n=sample_size, random_state=42)
    
    features_list = []
    
    logger.info(f"Processing {len(sampled_txns)} transactions...")
    
    for idx, (_, txn) in enumerate(sampled_txns.iterrows()):
        if idx % 2500 == 0:
            logger.info(f"Progress: {idx}/{len(sampled_txns)}")
        
        customer_id = txn['customer_id']
        
        # Find customer (with error handling)
        customer_rows = customers_df[customers_df['customer_id'] == customer_id]
        if len(customer_rows) == 0:
            continue
        customer = customer_rows.iloc[0]
        
        # Get customer transactions for behavioral features
        customer_txns = transactions_df[transactions_df['customer_id'] == customer_id]
        
        features = {
            # Transaction-level features (Contextual - Pillar 4)
            'transaction_amount': float(txn['amount']),
            'hour_of_day': int(txn['timestamp'].hour),
            'day_of_week': int(txn['timestamp'].weekday()),
            'is_weekend': int(txn['timestamp'].weekday() >= 5),
            'is_night_transaction': int(txn['timestamp'].hour >= 22 or txn['timestamp'].hour <= 6),
            'product_type_risk': get_product_risk_score(txn['product_type']),
            
            # Customer profile features (Profile - Pillar 1)
            'customer_age': int(customer['age']),
            'customer_income': float(customer['income']),
            'credit_score': int(customer['credit_score']),
            'account_age_days': int(customer['account_age_days']),
            'is_pep': int(customer['is_pep']),
            'high_risk_location': int(customer['high_risk_location']),
            'customer_risk_score': float(customer['risk_score']),
            
            # Behavioral features (Behavioral - Pillar 2)
            'avg_transaction_amount': float(customer_txns['amount'].mean()),
            'transaction_frequency': len(customer_txns),
            'unique_beneficiaries': int(customer_txns['beneficiary_id'].nunique()),
            'velocity_score': len(customer_txns) / max(customer['account_age_days'], 1) * 30,
            'weekend_transaction_ratio': sum(pd.to_datetime(customer_txns['timestamp']).dt.weekday >= 5) / len(customer_txns),
            'night_transaction_ratio': sum((pd.to_datetime(customer_txns['timestamp']).dt.hour >= 22) | 
                                         (pd.to_datetime(customer_txns['timestamp']).dt.hour <= 6)) / len(customer_txns),
            
            # Network features (Network - Pillar 3)
            'device_risk_score': get_device_risk_score(txn['device_id']),
            'merchant_risk_score': float(txn['location_risk']),  # Using location_risk as proxy
            'network_centrality': min(customer_txns['beneficiary_id'].nunique() / 100, 1.0),
            
            # Target
            'is_fraud': int(txn['is_fraud'])
        }
        
        features_list.append(features)
    
    # Create DataFrame
    features_df = pd.DataFrame(features_list)
    
    # Save features
    features_df.to_csv("data/demo/ml_features.csv", index=False)
    
    # Create summary
    summary = {
        'creation_time': datetime.now().isoformat(),
        'total_features': len(features_df),
        'fraud_rate': float(features_df['is_fraud'].mean()),
        'feature_columns': list(features_df.columns),
        'feature_statistics': {
            'avg_transaction_amount': float(features_df['transaction_amount'].mean()),
            'avg_customer_age': float(features_df['customer_age'].mean()),
            'avg_credit_score': float(features_df['credit_score'].mean()),
            'night_transaction_percentage': float(features_df['is_night_transaction'].mean() * 100),
            'weekend_transaction_percentage': float(features_df['is_weekend'].mean() * 100)
        }
    }
    
    import json
    with open("data/demo/ml_features_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ ML Features created successfully!")
    print(f"📊 Features: {len(features_df):,} samples")
    print(f"📊 Columns: {len(features_df.columns)} features")
    print(f"⚠️ Fraud rate: {features_df['is_fraud'].mean():.2%}")
    print(f"💾 Saved to: data/demo/ml_features.csv")
    
    return features_df

def get_product_risk_score(product_type):
    """Get risk score for product type"""
    risk_scores = {
        'PIX': 0.7,  # High risk
        'TED': 0.5,  # Medium risk  
        'DOC': 0.3,  # Lower risk
        'DEBIT': 0.2  # Lowest risk
    }
    return risk_scores.get(product_type, 0.5)

def get_device_risk_score(device_id):
    """Get risk score for device"""
    device_num = int(device_id.split('_')[1])
    if device_num > 9000:  # High-numbered devices are riskier
        return 0.8
    else:
        return 0.2

if __name__ == "__main__":
    create_ml_features()