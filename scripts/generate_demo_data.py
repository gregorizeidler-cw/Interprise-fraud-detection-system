#!/usr/bin/env python3
"""
Generate comprehensive demo datasets for fraud detection system

This script generates realistic synthetic datasets to demonstrate:
- Graph Neural Networks (transaction networks)
- SHAP Explainability (feature importance)
- Time Series Analysis (temporal patterns)
- Advanced fraud patterns and rings
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import networkx as nx
from typing import Tuple, List, Dict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DemoDataGenerator:
    """Generate comprehensive demo datasets for fraud detection"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.customers_pool = 10000
        self.devices_pool = 5000
        self.merchants_pool = 2000
        
    def generate_customer_profiles(self, n_customers: int = 10000) -> pd.DataFrame:
        """Generate customer profile data (4-Pillar Architecture - Pillar 1)"""
        
        logger.info(f"Generating {n_customers} customer profiles...")
        
        customers = []
        for i in range(n_customers):
            # Demographics
            age = max(18, int(np.random.normal(40, 15)))
            
            # Income based on age and education
            education_level = np.random.choice(['high_school', 'bachelor', 'master', 'phd'], 
                                             p=[0.3, 0.5, 0.15, 0.05])
            
            base_income = {
                'high_school': 30000, 'bachelor': 50000, 
                'master': 70000, 'phd': 90000
            }[education_level]
            
            income = max(20000, int(np.random.lognormal(np.log(base_income), 0.5)))
            
            # Credit score
            credit_score = int(np.random.beta(2, 2) * 850 + 300)
            
            # Account tenure
            account_age_days = np.random.exponential(365 * 3)  # Average 3 years
            
            # Risk factors
            is_pep = np.random.random() < 0.02  # 2% politically exposed persons
            is_high_risk_location = np.random.random() < 0.15  # 15% high-risk locations
            
            customer = {
                'customer_id': f"CUST_{i:08d}",
                'age': age,
                'income': income,
                'education': education_level,
                'credit_score': credit_score,
                'account_age_days': int(account_age_days),
                'is_pep': is_pep,
                'high_risk_location': is_high_risk_location,
                'registration_date': datetime.now() - timedelta(days=account_age_days),
                'customer_segment': self._determine_segment(income, credit_score),
                'risk_score': self._calculate_base_risk_score(age, income, credit_score, is_pep)
            }
            customers.append(customer)
        
        df = pd.DataFrame(customers)
        logger.info(f"✅ Generated {len(df)} customer profiles")
        return df
    
    def generate_transaction_network(self, 
                                   customers_df: pd.DataFrame,
                                   n_transactions: int = 50000,
                                   fraud_rate: float = 0.05) -> pd.DataFrame:
        """Generate transaction network data for GNN analysis"""
        
        logger.info(f"Generating {n_transactions} transactions with {fraud_rate:.1%} fraud rate...")
        
        customers = customers_df['customer_id'].tolist()
        transactions = []
        
        # Create fraud rings (highly connected subgraphs)
        fraud_rings = self._create_fraud_rings(customers, num_rings=5, ring_size=15)
        
        # Generate normal transactions
        normal_txns = int(n_transactions * (1 - fraud_rate))
        for i in range(normal_txns):
            customer = np.random.choice(customers)
            beneficiary = np.random.choice(customers)
            
            while beneficiary == customer:
                beneficiary = np.random.choice(customers)
            
            # Get customer info for realistic amounts
            cust_info = customers_df[customers_df['customer_id'] == customer].iloc[0]
            
            transaction = self._create_transaction(
                i, customer, beneficiary, cust_info, is_fraud=False
            )
            transactions.append(transaction)
        
        # Generate fraudulent transactions (including ring activity)
        fraud_txns = n_transactions - normal_txns
        for i in range(fraud_txns):
            # 70% chance of intra-ring fraud, 30% random fraud
            if np.random.random() < 0.7 and fraud_rings:
                ring = np.random.choice(fraud_rings)
                customer = np.random.choice(ring)
                beneficiary = np.random.choice(ring)
                
                while beneficiary == customer:
                    beneficiary = np.random.choice(ring)
            else:
                customer = np.random.choice(customers)
                beneficiary = np.random.choice(customers)
                while beneficiary == customer:
                    beneficiary = np.random.choice(customers)
            
            cust_info = customers_df[customers_df['customer_id'] == customer].iloc[0]
            
            transaction = self._create_transaction(
                normal_txns + i, customer, beneficiary, cust_info, is_fraud=True
            )
            transactions.append(transaction)
        
        df = pd.DataFrame(transactions)
        logger.info(f"✅ Generated {len(df)} transactions")
        logger.info(f"📊 Fraud transactions: {df['is_fraud'].sum()} ({df['is_fraud'].mean():.2%})")
        
        return df
    
    def generate_temporal_features_data(self, 
                                      transactions_df: pd.DataFrame,
                                      customers_df: pd.DataFrame) -> pd.DataFrame:
        """Generate time series data for temporal feature analysis"""
        
        logger.info("Generating temporal features dataset...")
        
        temporal_data = []
        
        for _, customer in customers_df.iterrows():
            customer_id = customer['customer_id']
            customer_txns = transactions_df[
                transactions_df['customer_id'] == customer_id
            ].sort_values('timestamp')
            
            if len(customer_txns) == 0:
                continue
            
            # Calculate temporal features
            features = self._extract_temporal_features(customer_txns, customer)
            features['customer_id'] = customer_id
            temporal_data.append(features)
        
        df = pd.DataFrame(temporal_data)
        logger.info(f"✅ Generated temporal features for {len(df)} customers")
        
        return df
    
    def generate_shap_features_data(self,
                                  transactions_df: pd.DataFrame,
                                  customers_df: pd.DataFrame,
                                  temporal_df: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive feature set for SHAP analysis"""
        
        logger.info("Generating SHAP features dataset...")
        
        # Merge all feature sources
        features_data = []
        
        for _, txn in transactions_df.iterrows():
            customer_id = txn['customer_id']
            
            # Get customer profile
            customer = customers_df[customers_df['customer_id'] == customer_id].iloc[0]
            
            # Get temporal features
            temporal = temporal_df[temporal_df['customer_id'] == customer_id]
            if len(temporal) == 0:
                continue
            temporal = temporal.iloc[0]
            
            # Combine all features
            features = {
                # Transaction-level features (Contextual - Pillar 4)
                'transaction_amount': txn['amount'],
                'hour_of_day': txn['timestamp'].hour,
                'day_of_week': txn['timestamp'].weekday(),
                'is_weekend': txn['timestamp'].weekday() >= 5,
                'is_night_transaction': txn['timestamp'].hour >= 22 or txn['timestamp'].hour <= 6,
                'product_type_risk': self._get_product_risk_score(txn['product_type']),
                
                # Customer profile features (Profile - Pillar 1)
                'customer_age': customer['age'],
                'customer_income': customer['income'],
                'credit_score': customer['credit_score'],
                'account_age_days': customer['account_age_days'],
                'is_pep': int(customer['is_pep']),
                'high_risk_location': int(customer['high_risk_location']),
                'customer_risk_score': customer['risk_score'],
                
                # Behavioral features (Behavioral - Pillar 2)
                'avg_transaction_amount': temporal['avg_transaction_amount'],
                'transaction_frequency': temporal['transaction_frequency'],
                'unique_beneficiaries': temporal['unique_beneficiaries'],
                'velocity_1h': temporal['velocity_1h'],
                'velocity_24h': temporal['velocity_24h'],
                'weekend_transaction_ratio': temporal['weekend_transaction_ratio'],
                'night_transaction_ratio': temporal['night_transaction_ratio'],
                
                # Network features (Network - Pillar 3)
                'device_risk_score': self._get_device_risk_score(txn['device_id']),
                'beneficiary_risk_score': self._get_beneficiary_risk_score(txn['beneficiary_id']),
                'network_centrality': temporal.get('network_centrality', 0),
                
                # Target
                'is_fraud': txn['is_fraud']
            }
            
            features_data.append(features)
        
        df = pd.DataFrame(features_data)
        logger.info(f"✅ Generated SHAP features for {len(df)} transactions")
        
        return df
    
    def _determine_segment(self, income: float, credit_score: int) -> str:
        """Determine customer segment based on income and credit score"""
        if income >= 100000 and credit_score >= 750:
            return 'premium'
        elif income >= 50000 and credit_score >= 650:
            return 'standard'
        else:
            return 'basic'
    
    def _calculate_base_risk_score(self, age: int, income: float, 
                                 credit_score: int, is_pep: bool) -> float:
        """Calculate base risk score for customer"""
        risk = 0.0
        
        # Age risk
        if age < 25 or age > 70:
            risk += 0.2
        
        # Income risk
        if income < 30000:
            risk += 0.3
        
        # Credit score risk
        if credit_score < 600:
            risk += 0.4
        elif credit_score < 700:
            risk += 0.2
        
        # PEP risk
        if is_pep:
            risk += 0.5
        
        return min(risk, 1.0)
    
    def _create_fraud_rings(self, customers: List[str], 
                          num_rings: int = 5, ring_size: int = 15) -> List[List[str]]:
        """Create fraud rings (connected groups of customers)"""
        rings = []
        used_customers = set()
        
        for _ in range(num_rings):
            available = [c for c in customers if c not in used_customers]
            if len(available) < ring_size:
                break
                
            ring = np.random.choice(available, ring_size, replace=False).tolist()
            rings.append(ring)
            used_customers.update(ring)
        
        return rings
    
    def _create_transaction(self, txn_id: int, customer: str, beneficiary: str,
                          customer_info: pd.Series, is_fraud: bool) -> Dict:
        """Create a single transaction"""
        
        # Base transaction amount based on customer profile
        base_amount = customer_info['income'] / 120  # Monthly income / 4
        
        if is_fraud:
            # Fraudulent transactions tend to be higher amounts
            amount = np.random.lognormal(np.log(base_amount * 3), 1.5)
            
            # Fraud more likely at night and weekends
            if np.random.random() < 0.4:  # 40% night transactions
                hour = np.random.randint(22, 24) if np.random.random() < 0.5 else np.random.randint(0, 6)
            else:
                hour = np.random.randint(0, 24)
                
            # Product type bias for fraud
            product_type = np.random.choice(['PIX', 'TED', 'DOC'], p=[0.6, 0.3, 0.1])
            
        else:
            # Normal transactions
            amount = np.random.lognormal(np.log(base_amount), 1.0)
            hour = np.random.choice(range(24), p=self._get_hourly_distribution())
            product_type = np.random.choice(['PIX', 'TED', 'DOC', 'DEBIT'], p=[0.4, 0.2, 0.2, 0.2])
        
        # Create timestamp
        days_ago = int(np.random.randint(0, 90))
        timestamp = datetime.now() - timedelta(
            days=days_ago,
            hours=int(hour),
            minutes=int(np.random.randint(0, 60)),
            seconds=int(np.random.randint(0, 60))
        )
        
        return {
            'transaction_id': f"{'FRAUD' if is_fraud else 'TXN'}_{txn_id:08d}",
            'customer_id': customer,
            'beneficiary_id': beneficiary,
            'amount': round(amount, 2),
            'product_type': product_type,
            'timestamp': timestamp,
            'device_id': f"DEV_{np.random.randint(1, self.devices_pool):06d}",
            'merchant_id': f"MERCH_{np.random.randint(1, self.merchants_pool):06d}",
            'channel': np.random.choice(['mobile', 'web', 'atm', 'branch'], p=[0.5, 0.3, 0.15, 0.05]),
            'location_risk': np.random.beta(1, 4),  # Most locations low risk
            'is_fraud': int(is_fraud)
        }
    
    def _get_hourly_distribution(self) -> List[float]:
        """Get realistic hourly transaction distribution"""
        # Higher activity during business hours
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
        
        # Normalize
        total = sum(hours)
        return [h/total for h in hours]
    
    def _extract_temporal_features(self, customer_txns: pd.DataFrame, 
                                 customer_info: pd.Series) -> Dict:
        """Extract temporal features for a customer"""
        
        if len(customer_txns) == 0:
            return {
                'avg_transaction_amount': 0,
                'transaction_frequency': 0,
                'unique_beneficiaries': 0,
                'velocity_1h': 0,
                'velocity_24h': 0,
                'weekend_transaction_ratio': 0,
                'night_transaction_ratio': 0,
                'network_centrality': 0
            }
        
        # Sort by timestamp
        customer_txns = customer_txns.sort_values('timestamp')
        
        # Basic statistics
        avg_amount = customer_txns['amount'].mean()
        frequency = len(customer_txns) / max((customer_txns['timestamp'].max() - 
                                            customer_txns['timestamp'].min()).days, 1)
        
        # Unique beneficiaries
        unique_beneficiaries = customer_txns['beneficiary_id'].nunique()
        
        # Velocity features (simplified)
        velocity_1h = len(customer_txns) / (24 * 90)  # Approximate hourly rate over 90 days
        velocity_24h = len(customer_txns) / 90  # Approximate daily rate
        
        # Time-based ratios
        weekend_ratio = sum(customer_txns['timestamp'].dt.weekday >= 5) / len(customer_txns)
        night_ratio = sum((customer_txns['timestamp'].dt.hour >= 22) | 
                         (customer_txns['timestamp'].dt.hour <= 6)) / len(customer_txns)
        
        # Simple network centrality (number of unique connections)
        network_centrality = unique_beneficiaries / 100  # Normalized
        
        return {
            'avg_transaction_amount': avg_amount,
            'transaction_frequency': frequency,
            'unique_beneficiaries': unique_beneficiaries,
            'velocity_1h': velocity_1h,
            'velocity_24h': velocity_24h,
            'weekend_transaction_ratio': weekend_ratio,
            'night_transaction_ratio': night_ratio,
            'network_centrality': network_centrality
        }
    
    def _get_product_risk_score(self, product_type: str) -> float:
        """Get risk score for product type"""
        risk_scores = {
            'PIX': 0.7,  # High risk - instant, hard to reverse
            'TED': 0.5,  # Medium risk
            'DOC': 0.3,  # Lower risk
            'DEBIT': 0.2  # Lowest risk
        }
        return risk_scores.get(product_type, 0.5)
    
    def _get_device_risk_score(self, device_id: str) -> float:
        """Get risk score for device"""
        # Some devices are higher risk (shared, compromised)
        device_num = int(device_id.split('_')[1])
        if device_num > 9000:  # High-numbered devices are riskier
            return np.random.beta(3, 2)  # Higher risk
        else:
            return np.random.beta(1, 4)  # Lower risk
    
    def _get_beneficiary_risk_score(self, beneficiary_id: str) -> float:
        """Get risk score for beneficiary"""
        # Some beneficiaries are higher risk
        return np.random.beta(1, 3)  # Most beneficiaries low risk


def main():
    """Generate all demo datasets"""
    
    print("🚀 Generating Demo Datasets for Enterprise Fraud Detection System")
    print("=" * 80)
    
    generator = DemoDataGenerator()
    
    # Create output directory
    output_dir = "data/demo"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Generate customer profiles
    print("\n📊 Step 1: Generating Customer Profiles...")
    customers_df = generator.generate_customer_profiles(n_customers=5000)
    customers_df.to_csv(f"{output_dir}/customer_profiles.csv", index=False)
    print(f"✅ Saved to {output_dir}/customer_profiles.csv")
    
    # 2. Generate transaction network
    print("\n🕸️ Step 2: Generating Transaction Network...")
    transactions_df = generator.generate_transaction_network(
        customers_df, n_transactions=25000, fraud_rate=0.05
    )
    transactions_df.to_csv(f"{output_dir}/transaction_network.csv", index=False)
    print(f"✅ Saved to {output_dir}/transaction_network.csv")
    
    # 3. Generate temporal features
    print("\n⏰ Step 3: Generating Temporal Features...")
    temporal_df = generator.generate_temporal_features_data(transactions_df, customers_df)
    temporal_df.to_csv(f"{output_dir}/temporal_features.csv", index=False)
    print(f"✅ Saved to {output_dir}/temporal_features.csv")
    
    # 4. Generate SHAP features
    print("\n🔍 Step 4: Generating SHAP Features...")
    shap_df = generator.generate_shap_features_data(transactions_df, customers_df, temporal_df)
    shap_df.to_csv(f"{output_dir}/shap_features.csv", index=False)
    print(f"✅ Saved to {output_dir}/shap_features.csv")
    
    # 5. Generate summary statistics
    print("\n📊 Step 5: Generating Summary Statistics...")
    summary = {
        'dataset_info': {
            'customers': len(customers_df),
            'transactions': len(transactions_df),
            'fraud_rate': transactions_df['is_fraud'].mean(),
            'date_range': f"{transactions_df['timestamp'].min()} to {transactions_df['timestamp'].max()}",
            'products': transactions_df['product_type'].unique().tolist(),
            'channels': transactions_df['channel'].unique().tolist()
        },
        'customer_segments': customers_df['customer_segment'].value_counts().to_dict(),
        'fraud_by_product': transactions_df.groupby('product_type')['is_fraud'].agg(['count', 'sum', 'mean']).to_dict(),
        'temporal_patterns': {
            'avg_daily_transactions': len(transactions_df) / 90,
            'peak_hour': transactions_df['timestamp'].dt.hour.mode().iloc[0],
            'weekend_ratio': sum(transactions_df['timestamp'].dt.weekday >= 5) / len(transactions_df)
        }
    }
    
    import json
    with open(f"{output_dir}/dataset_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"✅ Saved to {output_dir}/dataset_summary.json")
    
    print("\n🎉 Demo Datasets Generation Complete!")
    print(f"📁 All files saved to: {output_dir}/")
    print("\n📋 Generated Files:")
    print(f"   • customer_profiles.csv    - {len(customers_df):,} customer profiles")
    print(f"   • transaction_network.csv  - {len(transactions_df):,} transactions")
    print(f"   • temporal_features.csv    - {len(temporal_df):,} temporal feature sets")
    print(f"   • shap_features.csv        - {len(shap_df):,} SHAP-ready feature vectors")
    print(f"   • dataset_summary.json     - Complete dataset statistics")
    
    return summary


if __name__ == "__main__":
    summary = main()