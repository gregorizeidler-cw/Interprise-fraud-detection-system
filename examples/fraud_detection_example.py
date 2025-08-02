#!/usr/bin/env python3
"""
Enterprise Fraud Detection System - Complete Example

This script demonstrates how to use the fraud detection system programmatically,
including model training, real-time inference, and monitoring.

Usage:
    python examples/fraud_detection_example.py
"""

import sys
import os
import requests
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np

# Add src to path
sys.path.append('src')

# Import our modules
from utils.config_manager import ConfigManager
from features.feature_store import FeatureStore
from models.hub_model import HubModelManager
from models.spoke_models import SpokeModelManager
from inference.fraud_detection_engine import FraudDetectionEngine, FraudDetectionRequest
from data.schemas import ProductType

class FraudDetectionExample:
    """Complete example of fraud detection system usage"""
    
    def __init__(self):
        self.config = ConfigManager()
        self.api_base_url = "http://localhost:8000"
        
        print("🚀 Enterprise Fraud Detection System Example")
        print("=" * 50)
        print(f"Environment: {self.config.environment}")
        print(f"API URL: {self.api_base_url}")
        print(f"Timestamp: {datetime.now()}")
        
    def check_system_health(self):
        """Check if the system is healthy and ready"""
        
        print("\n🔍 Checking System Health")
        print("-" * 30)
        
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ System Status: {health_data['status']}")
                
                print("\n📊 Component Status:")
                for component, status in health_data['components'].items():
                    if isinstance(status, dict):
                        print(f"  {component}:")
                        for sub_comp, sub_status in status.items():
                            emoji = "✅" if sub_status == "loaded" else "❌"
                            print(f"    {emoji} {sub_comp}: {sub_status}")
                    else:
                        emoji = "✅" if status == "healthy" else "❌"
                        print(f"  {emoji} {component}: {status}")
                
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to API. Make sure the system is running:")
            print("   ./scripts/start_system.sh")
            return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def demonstrate_single_transaction(self):
        """Demonstrate single transaction fraud detection"""
        
        print("\n💳 Single Transaction Fraud Detection")
        print("-" * 40)
        
        # Sample transactions with different risk profiles
        transactions = [
            {
                "name": "Normal PIX Transfer",
                "data": {
                    "transaction_id": "demo_pix_001",
                    "customer_id": "cust_123456",
                    "product_type": "pix",
                    "amount": 250.00,
                    "currency": "BRL",
                    "channel": "mobile_app",
                    "timestamp": datetime.now().isoformat(),
                    "beneficiary_id": "benef_familiar"
                }
            },
            {
                "name": "High-Value Credit Card",
                "data": {
                    "transaction_id": "demo_cc_002", 
                    "customer_id": "cust_789012",
                    "product_type": "credit_card",
                    "amount": 15000.00,
                    "currency": "BRL",
                    "channel": "web_browser",
                    "timestamp": datetime.now().isoformat(),
                    "merchant_category": "5732",  # Electronics
                    "merchant_id": "merchant_electronics_xyz"
                }
            },
            {
                "name": "Suspicious TED Transfer",
                "data": {
                    "transaction_id": "demo_ted_003",
                    "customer_id": "cust_345678",
                    "product_type": "ted", 
                    "amount": 50000.00,
                    "currency": "BRL",
                    "channel": "atm",
                    "timestamp": datetime.now().isoformat(),
                    "beneficiary_id": "benef_unknown_entity"
                }
            }
        ]
        
        for transaction in transactions:
            print(f"\n📋 {transaction['name']}")
            print(f"   Amount: R$ {transaction['data']['amount']:,.2f}")
            print(f"   Product: {transaction['data']['product_type'].upper()}")
            print(f"   Channel: {transaction['data']['channel']}")
            
            try:
                start_time = time.time()
                
                response = requests.post(
                    f"{self.api_base_url}/fraud-detection",
                    json=transaction['data'],
                    params={"include_explanation": True},
                    timeout=30
                )
                
                processing_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    result = response.json()
                    
                    print(f"   📊 Results:")
                    print(f"     Final Score: {result['final_score']:.3f}")
                    print(f"     Risk Level: {result['risk_level'].upper()}")
                    print(f"     Action: {result['action'].upper()}")
                    print(f"     Processing Time: {processing_time:.1f}ms")
                    
                    if result.get('reason_codes'):
                        print(f"     Reason Codes: {', '.join(result['reason_codes'])}")
                    
                    if result.get('explanation'):
                        print(f"     Summary: {result['explanation']['summary']}")
                        
                else:
                    print(f"   ❌ Error: {response.status_code} - {response.text}")
                    
            except Exception as e:
                print(f"   ❌ Request failed: {e}")
    
    def demonstrate_batch_processing(self):
        """Demonstrate batch transaction processing"""
        
        print("\n📦 Batch Transaction Processing")
        print("-" * 35)
        
        # Generate batch of test transactions
        batch_transactions = []
        product_types = ["pix", "credit_card", "ted"]
        channels = ["mobile_app", "web_browser", "atm"]
        
        for i in range(10):
            amount = np.random.lognormal(6, 1.5)  # Realistic amount distribution
            
            transaction = {
                "transaction_id": f"batch_demo_{i:03d}",
                "customer_id": f"cust_{np.random.randint(100000, 999999)}",
                "product_type": np.random.choice(product_types),
                "amount": round(amount, 2),
                "currency": "BRL",
                "channel": np.random.choice(channels),
                "timestamp": datetime.now().isoformat()
            }
            
            # Add product-specific fields
            if transaction["product_type"] in ["pix", "ted"]:
                transaction["beneficiary_id"] = f"benef_{np.random.randint(1000, 9999)}"
            elif transaction["product_type"] == "credit_card":
                transaction["merchant_id"] = f"merchant_{np.random.randint(100, 999)}"
                transaction["merchant_category"] = np.random.choice(["5411", "5812", "5999"])
            
            batch_transactions.append(transaction)
        
        print(f"📊 Processing {len(batch_transactions)} transactions in batch...")
        
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{self.api_base_url}/fraud-detection/batch",
                json={
                    "transactions": batch_transactions,
                    "include_explanation": False
                },
                timeout=60
            )
            
            total_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                results = response.json()
                
                # Analyze results
                risk_levels = [r['risk_level'] for r in results]
                actions = [r['action'] for r in results]
                processing_times = [r['processing_time_ms'] for r in results]
                
                print(f"\n✅ Batch processing completed in {total_time:.0f}ms")
                print(f"📊 Results Summary:")
                print(f"   Total Transactions: {len(results)}")
                print(f"   Average Processing Time: {np.mean(processing_times):.1f}ms per transaction")
                
                print(f"\n   Risk Distribution:")
                for level in ['low', 'medium', 'high']:
                    count = risk_levels.count(level)
                    pct = (count / len(risk_levels)) * 100
                    print(f"     {level.capitalize()}: {count} ({pct:.1f}%)")
                
                print(f"\n   Action Distribution:")
                for action in ['approve', 'challenge', 'reject']:
                    count = actions.count(action)
                    pct = (count / len(actions)) * 100
                    print(f"     {action.capitalize()}: {count} ({pct:.1f}%)")
                    
            else:
                print(f"❌ Batch processing failed: {response.status_code}")
                print(response.text)
                
        except Exception as e:
            print(f"❌ Batch processing error: {e}")
    
    def check_system_metrics(self):
        """Check system performance metrics"""
        
        print("\n📈 System Performance Metrics")
        print("-" * 35)
        
        try:
            response = requests.get(f"{self.api_base_url}/metrics", timeout=10)
            
            if response.status_code == 200:
                metrics = response.json()
                
                print(f"📊 Performance Statistics:")
                print(f"   Total Requests: {metrics['total_requests']:,}")
                print(f"   Average Response Time: {metrics['average_processing_time_ms']:.1f}ms")
                print(f"   Requests per Second: {metrics['requests_per_second']:.1f}")
                print(f"   Last Updated: {metrics['timestamp']}")
                
            else:
                print(f"❌ Cannot retrieve metrics: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Metrics error: {e}")
    
    def demonstrate_model_info(self):
        """Show model information and versions"""
        
        print("\n🧠 Model Information")
        print("-" * 25)
        
        try:
            response = requests.get(f"{self.api_base_url}/models/info", timeout=10)
            
            if response.status_code == 200:
                model_info = response.json()
                
                print(f"🎯 Hub Model:")
                hub_info = model_info.get('hub_model', {})
                print(f"   Status: {hub_info.get('status', 'Unknown')}")
                
                if 'metadata' in hub_info:
                    metadata = hub_info['metadata']
                    print(f"   Algorithm: {metadata.get('algorithm', 'Unknown')}")
                    print(f"   Version: {metadata.get('version', 'Unknown')}")
                    print(f"   Trained: {metadata.get('trained_at', 'Unknown')}")
                
                print(f"\n🎛️ Spoke Models:")
                spoke_info = model_info.get('spoke_models', {})
                for product, info in spoke_info.items():
                    status = info.get('status', 'Unknown')
                    print(f"   {product.upper()}: {status}")
                    
            else:
                print(f"❌ Cannot retrieve model info: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Model info error: {e}")
    
    def run_complete_example(self):
        """Run the complete fraud detection example"""
        
        print("\n🎯 Running Complete Fraud Detection Example")
        print("=" * 50)
        
        # 1. Check system health
        if not self.check_system_health():
            print("\n❌ System is not healthy. Please start the system first:")
            print("   ./scripts/start_system.sh")
            return False
        
        # 2. Show model information
        self.demonstrate_model_info()
        
        # 3. Demonstrate single transactions
        self.demonstrate_single_transaction()
        
        # 4. Demonstrate batch processing
        self.demonstrate_batch_processing()
        
        # 5. Check performance metrics
        self.check_system_metrics()
        
        print("\n🎉 Complete example finished successfully!")
        print("\n📚 Next Steps:")
        print("   - Explore Jupyter notebooks in notebooks/")
        print("   - Check API documentation at http://localhost:8000/docs")
        print("   - Monitor system health at http://localhost:8000/health")
        print("   - View system metrics at http://localhost:8000/metrics")
        
        return True

def main():
    """Main function"""
    
    example = FraudDetectionExample()
    
    try:
        success = example.run_complete_example()
        
        if success:
            print("\n✨ Example completed successfully!")
        else:
            print("\n❌ Example failed. Check the system status.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Example interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()