#!/usr/bin/env python3
"""
Fast Demo Data Generator for Enterprise Fraud Detection System
Optimized version for generating 100,000+ transactions quickly
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FastDemoDataGenerator:
    """Optimized demo data generator"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        
    def generate_customer_profiles(self, n_customers: int = 10000) -> pd.DataFrame:
        """Generate customer profiles efficiently"""
        
        logger.info(f"Generating {n_customers} customer profiles...")
        
        # Generate all data at once using vectorized operations
        ages = np.maximum(18, np.random.normal(40, 15, n_customers).astype(int))
        
        # Education levels
        education_choices = ['high_school', 'bachelor', 'master', 'phd']
        education_probs = [0.3, 0.5, 0.15, 0.05]
        educations = np.random.choice(education_choices, n_customers, p=education_probs)
        
        # Income based on education
        base_incomes = np.select([
            educations == 'high_school',
            educations == 'bachelor', 
            educations == 'master',
            educations == 'phd'
        ], [30000, 50000, 70000, 90000], default=50000)
        
        incomes = np.maximum(20000, np.random.lognormal(np.log(base_incomes), 0.5))
        
        # Credit scores
        credit_scores = (np.random.beta(2, 2, n_customers) * 550 + 300).astype(int)
        
        # Account ages
        account_ages = np.random.exponential(365 * 3, n_customers).astype(int)
        
        # Risk flags
        is_pep = np.random.random(n_customers) < 0.02
        is_high_risk = np.random.random(n_customers) < 0.15
        
        # Create DataFrame
        df = pd.DataFrame({
            'customer_id': [f"CUST_{i:08d}" for i in range(n_customers)],
            'age': ages,
            'income': incomes.astype(int),
            'education': educations,
            'credit_score': credit_scores,
            'account_age_days': account_ages,
            'is_pep': is_pep,
            'high_risk_location': is_high_risk,
            'registration_date': [datetime.now() - timedelta(days=int(age)) for age in account_ages]
        })
        
        # Add derived fields
        df['customer_segment'] = np.select([
            (df['income'] >= 100000) & (df['credit_score'] >= 750),
            (df['income'] >= 50000) & (df['credit_score'] >= 650)
        ], ['premium', 'standard'], default='basic')
        
        # Calculate risk scores
        risk_scores = np.zeros(n_customers)
        risk_scores += np.where((df['age'] < 25) | (df['age'] > 70), 0.2, 0)
        risk_scores += np.where(df['income'] < 30000, 0.3, 0)
        risk_scores += np.where(df['credit_score'] < 600, 0.4, 
                              np.where(df['credit_score'] < 700, 0.2, 0))
        risk_scores += np.where(df['is_pep'], 0.5, 0)
        df['risk_score'] = np.minimum(risk_scores, 1.0)
        
        logger.info(f"✅ Generated {len(df)} customer profiles")
        return df
    
    def generate_transaction_network(self, customers_df: pd.DataFrame, 
                                   n_transactions: int = 100000,
                                   fraud_rate: float = 0.05) -> pd.DataFrame:
        """Generate transactions efficiently"""
        
        logger.info(f"Generating {n_transactions} transactions with {fraud_rate:.1%} fraud rate...")
        
        customers = customers_df['customer_id'].values
        n_customers = len(customers)
        
        # Pre-allocate arrays for efficiency
        transaction_ids = []
        customer_ids = np.empty(n_transactions, dtype='U12')
        beneficiary_ids = np.empty(n_transactions, dtype='U12')
        amounts = np.empty(n_transactions)
        product_types = np.empty(n_transactions, dtype='U12')
        timestamps = np.empty(n_transactions, dtype='datetime64[ns]')
        device_ids = np.empty(n_transactions, dtype='U10')
        merchant_ids = np.empty(n_transactions, dtype='U12')
        channels = np.empty(n_transactions, dtype='U10')
        location_risks = np.empty(n_transactions)
        is_fraud = np.zeros(n_transactions, dtype=int)
        
        # Create fraud rings
        fraud_ring_customers = self._create_fraud_rings_fast(customers, 5, 15)
        
        # Generate normal transactions
        n_normal = int(n_transactions * (1 - fraud_rate))
        
        # Batch generate normal transactions
        logger.info("Generating normal transactions...")
        normal_customers = np.random.choice(customers, n_normal)
        normal_beneficiaries = np.random.choice(customers, n_normal)
        
        # Ensure no self-transactions
        self_txn_mask = normal_customers == normal_beneficiaries
        while self_txn_mask.any():
            normal_beneficiaries[self_txn_mask] = np.random.choice(customers, self_txn_mask.sum())
            self_txn_mask = normal_customers == normal_beneficiaries
        
        customer_ids[:n_normal] = normal_customers
        beneficiary_ids[:n_normal] = normal_beneficiaries
        
        # Generate amounts based on customer income
        for i in range(n_normal):
            if i % 10000 == 0:
                logger.info(f"Processing normal transaction {i}/{n_normal}...")
            customer_income = customers_df[customers_df['customer_id'] == normal_customers[i]]['income'].iloc[0]
            base_amount = customer_income / 120
            amounts[i] = np.random.lognormal(np.log(base_amount), 1.0)
        
        # Generate other normal transaction attributes
        product_choices = ['PIX', 'TED', 'DOC', 'DEBIT']
        product_probs = [0.4, 0.2, 0.2, 0.2]
        product_types[:n_normal] = np.random.choice(product_choices, n_normal, p=product_probs)
        
        # Generate timestamps
        base_time = datetime.now()
        for i in range(n_normal):
            days_ago = np.random.randint(0, 90)
            hour = np.random.choice(range(24), p=self._get_hourly_distribution())
            timestamps[i] = base_time - timedelta(
                days=int(days_ago), 
                hours=int(hour),
                minutes=int(np.random.randint(0, 60))
            )
        
        # Generate fraud transactions
        logger.info("Generating fraudulent transactions...")
        n_fraud = n_transactions - n_normal
        
        # 70% intra-ring fraud, 30% random fraud
        intra_ring_count = int(n_fraud * 0.7)
        random_fraud_count = n_fraud - intra_ring_count
        
        fraud_start_idx = n_normal
        
        # Intra-ring fraud
        if fraud_ring_customers and intra_ring_count > 0:
            for i in range(intra_ring_count):
                idx = fraud_start_idx + i
                ring = np.random.choice(len(fraud_ring_customers))
                ring_customers = fraud_ring_customers[ring]
                
                customer = np.random.choice(ring_customers)
                beneficiary = np.random.choice(ring_customers)
                while beneficiary == customer:
                    beneficiary = np.random.choice(ring_customers)
                
                customer_ids[idx] = customer
                beneficiary_ids[idx] = beneficiary
                
                # Higher amounts for fraud
                customer_income = customers_df[customers_df['customer_id'] == customer]['income'].iloc[0]
                base_amount = customer_income / 40  # Higher than normal
                amounts[idx] = np.random.lognormal(np.log(base_amount), 1.5)
                
                is_fraud[idx] = 1
        
        # Random fraud
        random_start = fraud_start_idx + intra_ring_count
        for i in range(random_fraud_count):
            idx = random_start + i
            customer = np.random.choice(customers)
            beneficiary = np.random.choice(customers)
            while beneficiary == customer:
                beneficiary = np.random.choice(customers)
            
            customer_ids[idx] = customer
            beneficiary_ids[idx] = beneficiary
            
            # Higher amounts for fraud
            customer_income = customers_df[customers_df['customer_id'] == customer]['income'].iloc[0]
            base_amount = customer_income / 40
            amounts[idx] = np.random.lognormal(np.log(base_amount), 1.5)
            
            is_fraud[idx] = 1
        
        # Generate fraud product types (bias towards PIX/TED)
        fraud_products = np.random.choice(['PIX', 'TED'], n_fraud, p=[0.8, 0.2])
        product_types[n_normal:] = fraud_products
        
        # Generate fraud timestamps (bias towards night)
        for i in range(n_normal, n_transactions):
            days_ago = np.random.randint(0, 30)  # More recent
            if np.random.random() < 0.4:  # 40% night transactions
                hour = np.random.choice([22, 23, 0, 1, 2, 3, 4, 5])
            else:
                hour = np.random.randint(0, 24)
            
            timestamps[i] = base_time - timedelta(
                days=int(days_ago),
                hours=int(hour),
                minutes=int(np.random.randint(0, 60))
            )
        
        # Generate remaining attributes
        device_ids[:] = [f"DEV_{np.random.randint(1, 10000):06d}" for _ in range(n_transactions)]
        merchant_ids[:] = [f"MERCH_{np.random.randint(1, 2000):06d}" for _ in range(n_transactions)]
        channels[:] = np.random.choice(['mobile', 'web', 'atm', 'branch'], n_transactions, p=[0.5, 0.3, 0.15, 0.05])
        location_risks[:] = np.random.beta(1, 4, n_transactions)
        
        # Generate transaction IDs
        transaction_ids = [f"{'FRAUD' if is_fraud[i] else 'TXN'}_{i:08d}" for i in range(n_transactions)]
        
        # Create DataFrame
        df = pd.DataFrame({
            'transaction_id': transaction_ids,
            'customer_id': customer_ids,
            'beneficiary_id': beneficiary_ids,
            'amount': amounts,
            'product_type': product_types,
            'timestamp': timestamps,
            'device_id': device_ids,
            'merchant_id': merchant_ids,
            'channel': channels,
            'location_risk': location_risks,
            'is_fraud': is_fraud
        })
        
        logger.info(f"✅ Generated {len(df)} transactions")
        logger.info(f"📊 Fraud transactions: {df['is_fraud'].sum()} ({df['is_fraud'].mean():.2%})")
        
        return df
    
    def _create_fraud_rings_fast(self, customers, num_rings=5, ring_size=15):
        """Create fraud rings efficiently"""
        rings = []
        used_customers = set()
        
        for _ in range(num_rings):
            available = [c for c in customers if c not in used_customers]
            if len(available) < ring_size:
                break
            
            ring = np.random.choice(available, ring_size, replace=False)
            rings.append(ring)
            used_customers.update(ring)
        
        return rings
    
    def _get_hourly_distribution(self):
        """Get realistic hourly transaction distribution"""
        hours = []
        for h in range(24):
            if 9 <= h <= 17:  # Business hours
                hours.append(0.06)
            elif 18 <= h <= 21:  # Evening
                hours.append(0.04)
            elif 6 <= h <= 8:  # Morning
                hours.append(0.03)
            else:  # Night/early morning
                hours.append(0.01)
        
        total = sum(hours)
        return [h/total for h in hours]
    
    def generate_features_fast(self, transactions_df, customers_df):
        """Generate comprehensive features efficiently"""
        
        logger.info("Generating comprehensive feature set...")
        
        # Sample transactions for feature generation (to speed up)
        sample_size = min(50000, len(transactions_df))
        sampled_txns = transactions_df.sample(n=sample_size, random_state=42)
        
        features_list = []
        
        for idx, (_, txn) in enumerate(sampled_txns.iterrows()):
            if idx % 5000 == 0:
                logger.info(f"Processing transaction {idx}/{len(sampled_txns)}...")
            
            customer_id = txn['customer_id']
            customer = customers_df[customers_df['customer_id'] == customer_id].iloc[0]
            
            # Calculate customer behavioral features
            customer_txns = transactions_df[transactions_df['customer_id'] == customer_id]
            
            features = {
                # Transaction features
                'transaction_amount': txn['amount'],
                'hour_of_day': txn['timestamp'].hour,
                'day_of_week': txn['timestamp'].weekday(),
                'is_weekend': int(txn['timestamp'].weekday() >= 5),
                'is_night': int(txn['timestamp'].hour >= 22 or txn['timestamp'].hour <= 6),
                'product_risk': self._get_product_risk(txn['product_type']),
                
                # Customer profile
                'customer_age': customer['age'],
                'customer_income': customer['income'],
                'credit_score': customer['credit_score'],
                'account_age_days': customer['account_age_days'],
                'is_pep': int(customer['is_pep']),
                'high_risk_location': int(customer['high_risk_location']),
                'customer_risk_score': customer['risk_score'],
                
                # Behavioral features
                'avg_transaction_amount': customer_txns['amount'].mean(),
                'transaction_count': len(customer_txns),
                'unique_beneficiaries': customer_txns['beneficiary_id'].nunique(),
                'velocity_score': len(customer_txns) / max(customer['account_age_days'], 1),
                'weekend_ratio': sum(customer_txns['timestamp'].dt.weekday >= 5) / len(customer_txns),
                'night_ratio': sum((customer_txns['timestamp'].dt.hour >= 22) | 
                                 (customer_txns['timestamp'].dt.hour <= 6)) / len(customer_txns),
                
                # Network features (simplified)
                'device_risk': self._get_device_risk(txn['device_id']),
                'merchant_risk': np.random.beta(1, 3),  # Simplified
                'location_risk': txn['location_risk'],
                
                # Target
                'is_fraud': txn['is_fraud']
            }
            
            features_list.append(features)
        
        df = pd.DataFrame(features_list)
        logger.info(f"✅ Generated features for {len(df)} transactions")
        
        return df
    
    def _get_product_risk(self, product_type):
        """Get product risk score"""
        risks = {'PIX': 0.7, 'TED': 0.5, 'DOC': 0.3, 'DEBIT': 0.2}
        return risks.get(product_type, 0.5)
    
    def _get_device_risk(self, device_id):
        """Get device risk score"""
        device_num = int(device_id.split('_')[1])
        return 0.8 if device_num > 9000 else 0.2


def main():
    """Generate demo datasets efficiently"""
    
    print("🚀 Fast Demo Data Generation - Enterprise Fraud Detection")
    print("=" * 65)
    
    generator = FastDemoDataGenerator()
    output_dir = "data/demo"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Customer profiles
    print("\n👥 Step 1: Customer Profiles (10,000)...")
    customers_df = generator.generate_customer_profiles(10000)
    customers_df.to_csv(f"{output_dir}/customer_profiles.csv", index=False)
    print(f"✅ Saved: {output_dir}/customer_profiles.csv")
    
    # 2. Transaction network
    print("\n💰 Step 2: Transaction Network (100,000)...")
    transactions_df = generator.generate_transaction_network(customers_df, 100000, 0.05)
    transactions_df.to_csv(f"{output_dir}/transaction_network.csv", index=False)
    print(f"✅ Saved: {output_dir}/transaction_network.csv")
    
    # 3. Feature set for ML
    print("\n🔍 Step 3: ML Feature Set (50,000 samples)...")
    features_df = generator.generate_features_fast(transactions_df, customers_df)
    features_df.to_csv(f"{output_dir}/ml_features.csv", index=False)
    print(f"✅ Saved: {output_dir}/ml_features.csv")
    
    # 4. Summary statistics
    print("\n📊 Step 4: Summary Statistics...")
    summary = {
        'generation_time': datetime.now().isoformat(),
        'customers': len(customers_df),
        'transactions': len(transactions_df),
        'ml_features': len(features_df),
        'fraud_rate': float(transactions_df['is_fraud'].mean()),
        'customer_segments': customers_df['customer_segment'].value_counts().to_dict(),
        'product_distribution': transactions_df['product_type'].value_counts().to_dict(),
        'channel_distribution': transactions_df['channel'].value_counts().to_dict()
    }
    
    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Saved: {output_dir}/summary.json")
    
    print("\n🎉 Demo Data Generation Complete!")
    print(f"📂 Location: {output_dir}/")
    print(f"📊 {len(customers_df):,} customers")
    print(f"💰 {len(transactions_df):,} transactions")
    print(f"🔍 {len(features_df):,} ML features")
    print(f"⚠️  {transactions_df['is_fraud'].sum():,} fraudulent transactions ({transactions_df['is_fraud'].mean():.2%})")
    
    return summary


if __name__ == "__main__":
    summary = main()