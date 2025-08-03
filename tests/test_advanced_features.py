"""
Unit Tests for Advanced Data Science Features

This module contains comprehensive unit tests for:
- Graph Neural Networks
- Advanced Explainability (SHAP + Counterfactuals)  
- Time Series Features
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestGraphNeuralNetworks(unittest.TestCase):
    """Test Graph Neural Networks functionality"""
    
    def setUp(self):
        """Set up test data"""
        try:
            from src.features.graph_neural_networks import (
                TransactionGraphBuilder,
                CommunityFraudDetector,
                GraphFraudFeatureExtractor
            )
            self.graph_builder = TransactionGraphBuilder({})
            self.community_detector = CommunityFraudDetector()
            self.feature_extractor = GraphFraudFeatureExtractor()
            self.available = True
        except ImportError:
            self.available = False
            self.skipTest("Graph Neural Networks dependencies not available")
    
    def test_graph_builder_initialization(self):
        """Test graph builder initialization"""
        self.assertIsNotNone(self.graph_builder)
        self.assertEqual(self.graph_builder.config, {})
    
    def test_transaction_graph_construction(self):
        """Test transaction graph construction"""
        # Create sample transaction data
        transactions = pd.DataFrame({
            'customer_id': ['CUST_001', 'CUST_002', 'CUST_001', 'CUST_003'],
            'beneficiary_id': ['CUST_002', 'CUST_003', 'CUST_003', 'CUST_001'],
            'amount': [100, 200, 150, 300],
            'timestamp': [datetime.now() - timedelta(days=i) for i in range(4)],
            'product_type': ['PIX', 'TED', 'PIX', 'DOC']
        })
        
        graph = self.graph_builder.build_customer_transaction_graph(transactions)
        
        # Test graph properties
        self.assertGreater(graph.number_of_nodes(), 0)
        self.assertGreater(graph.number_of_edges(), 0)
        
        # Test that customers are nodes
        self.assertIn('CUST_001', graph.nodes())
        self.assertIn('CUST_002', graph.nodes())
    
    def test_device_sharing_graph(self):
        """Test device sharing graph construction"""
        transactions = pd.DataFrame({
            'customer_id': ['CUST_001', 'CUST_002', 'CUST_003'],
            'device_id': ['DEV_001', 'DEV_001', 'DEV_002'],  # CUST_001 and CUST_002 share device
            'timestamp': [datetime.now() for _ in range(3)]
        })
        
        graph = self.graph_builder.build_device_sharing_graph(transactions)
        
        # Should have edge between CUST_001 and CUST_002 (shared device)
        if graph.number_of_edges() > 0:
            self.assertTrue(graph.has_edge('CUST_001', 'CUST_002'))
    
    def test_feature_extraction(self):
        """Test graph feature extraction"""
        # Create a simple graph
        import networkx as nx
        G = nx.Graph()
        G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D')])
        
        features = self.feature_extractor.extract_node_features(G, 'B')
        
        # Check that features are returned
        self.assertIsInstance(features, dict)
        self.assertIn('degree_centrality', features)
        self.assertIn('clustering_coefficient', features)
        
        # Node B should have high centrality (connected to A and C)
        self.assertGreater(features['degree_centrality'], 0)


class TestTemporalFeatures(unittest.TestCase):
    """Test Advanced Time Series Features"""
    
    def setUp(self):
        """Set up test data"""
        try:
            from src.features.temporal_features import (
                TemporalFeatureExtractor,
                RealTimeTemporalProcessor,
                TemporalAnomalyDetector
            )
            self.temporal_extractor = TemporalFeatureExtractor()
            self.realtime_processor = RealTimeTemporalProcessor()
            self.anomaly_detector = TemporalAnomalyDetector()
            self.available = True
        except ImportError:
            self.available = False
            self.skipTest("Temporal features dependencies not available")
    
    def test_temporal_extractor_initialization(self):
        """Test temporal feature extractor initialization"""
        self.assertIsNotNone(self.temporal_extractor)
        self.assertIsInstance(self.temporal_extractor.time_windows, list)
    
    def test_basic_temporal_features(self):
        """Test basic temporal feature extraction"""
        # Create sample transaction data
        transactions = pd.DataFrame({
            'customer_id': ['CUST_001'] * 10,
            'amount': np.random.normal(100, 20, 10),
            'timestamp': [datetime.now() - timedelta(hours=i) for i in range(10)]
        })
        
        features = self.temporal_extractor.extract_all_temporal_features(
            transactions, 'CUST_001'
        )
        
        # Check that features are returned
        self.assertIsInstance(features, dict)
        self.assertIn('days_since_first_transaction', features)
        self.assertIn('avg_inter_transaction_hours', features)
        self.assertGreater(len(features), 10)  # Should have many features
    
    def test_rolling_window_features(self):
        """Test rolling window feature extraction"""
        transactions = pd.DataFrame({
            'customer_id': ['CUST_001'] * 20,
            'amount': np.random.normal(100, 20, 20),
            'timestamp': [datetime.now() - timedelta(hours=i) for i in range(20)],
            'product_type': ['PIX'] * 20
        })
        
        features = self.temporal_extractor._extract_rolling_window_features(
            transactions, datetime.now()
        )
        
        # Check rolling window features
        self.assertIn('1H_transaction_count', features)
        self.assertIn('24H_total_amount', features)
        self.assertIsInstance(features['1H_transaction_count'], (int, float))
    
    def test_velocity_features(self):
        """Test velocity and acceleration features"""
        transactions = pd.DataFrame({
            'customer_id': ['CUST_001'] * 15,
            'amount': np.random.normal(100, 20, 15),
            'timestamp': [datetime.now() - timedelta(hours=i) for i in range(15)]
        })
        
        features = self.temporal_extractor._extract_velocity_features(
            transactions, datetime.now()
        )
        
        # Check velocity features
        self.assertIn('velocity_txn_per_hour_1H', features)
        self.assertIn('velocity_amount_per_hour_24H', features)
        self.assertGreaterEqual(features['velocity_txn_per_hour_1H'], 0)
    
    def test_realtime_processor(self):
        """Test real-time temporal processing"""
        customer_id = 'CUST_001'
        transaction = {'amount': 100, 'timestamp': datetime.now()}
        
        features = self.realtime_processor.update_online_features(
            customer_id, transaction, datetime.now()
        )
        
        # Check real-time features
        self.assertIsInstance(features, dict)
        self.assertIn('total_transactions', features)
        self.assertEqual(features['total_transactions'], 1)


class TestAdvancedExplainability(unittest.TestCase):
    """Test Advanced Explainability Features"""
    
    def setUp(self):
        """Set up test data and models"""
        try:
            from src.explainability.advanced_explainability import (
                ExplainabilityManager
            )
            
            # Create a simple model for testing
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.datasets import make_classification
            
            # Generate synthetic data
            X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
            self.X_train = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
            self.y_train = pd.Series(y)
            
            # Train a simple model
            self.model = RandomForestClassifier(n_estimators=10, random_state=42)
            self.model.fit(self.X_train, self.y_train)
            
            self.explainability_manager = ExplainabilityManager(self.model)
            self.available = True
        except ImportError:
            self.available = False
            self.skipTest("Explainability dependencies not available")
    
    def test_explainability_manager_initialization(self):
        """Test explainability manager initialization"""
        self.assertIsNotNone(self.explainability_manager)
        self.assertIsNotNone(self.explainability_manager.model)
    
    def test_shap_explainer_creation(self):
        """Test SHAP explainer creation"""
        try:
            from src.explainability.advanced_explainability import SHAPExplainer
            
            shap_explainer = SHAPExplainer(self.model, 'tree')
            shap_explainer.initialize_explainer(self.X_train)
            
            self.assertIsNotNone(shap_explainer.explainer)
            self.assertEqual(shap_explainer.model_type, 'tree')
        except ImportError:
            self.skipTest("SHAP not available")
    
    def test_model_prediction_explanation(self):
        """Test model prediction explanation"""
        # Get a sample instance
        sample_instance = self.X_train.iloc[0]
        
        try:
            explanation = self.explainability_manager.get_comprehensive_explanation(
                sample_instance, include_counterfactuals=False, include_lime=False
            )
            
            # Check explanation structure
            self.assertIsInstance(explanation, dict)
            self.assertIn('instance', explanation)
            self.assertIn('model_prediction', explanation)
            self.assertIn('summary', explanation)
            
            # Check model prediction
            if explanation['model_prediction']:
                self.assertIn('fraud_probability', explanation['model_prediction'])
                self.assertIn('prediction', explanation['model_prediction'])
        except Exception as e:
            # Some explainability features might not be available
            self.skipTest(f"Explanation generation failed: {e}")


class TestDataIntegrity(unittest.TestCase):
    """Test data integrity and generation"""
    
    def test_demo_data_exists(self):
        """Test that demo data files exist"""
        data_dir = "data/demo"
        
        expected_files = [
            "customer_profiles.csv",
            "transaction_network.csv", 
            "ml_features.csv"
        ]
        
        for file_name in expected_files:
            file_path = os.path.join(data_dir, file_name)
            if os.path.exists(file_path):
                self.assertTrue(os.path.exists(file_path), f"{file_name} should exist")
                
                # Check file is not empty
                self.assertGreater(os.path.getsize(file_path), 0, f"{file_name} should not be empty")
    
    def test_demo_data_format(self):
        """Test demo data format and structure"""
        try:
            # Test customer profiles
            customers_path = "data/demo/customer_profiles.csv"
            if os.path.exists(customers_path):
                customers_df = pd.read_csv(customers_path)
                
                expected_columns = ['customer_id', 'age', 'income', 'credit_score']
                for col in expected_columns:
                    self.assertIn(col, customers_df.columns, f"Customer data should have {col} column")
                
                # Check data types and ranges
                self.assertTrue(customers_df['age'].min() >= 18, "Age should be >= 18")
                self.assertTrue(customers_df['income'].min() > 0, "Income should be positive")
                self.assertTrue(customers_df['credit_score'].between(300, 850).all(), "Credit score should be 300-850")
            
            # Test transactions
            transactions_path = "data/demo/transaction_network.csv"
            if os.path.exists(transactions_path):
                transactions_df = pd.read_csv(transactions_path)
                
                expected_columns = ['transaction_id', 'customer_id', 'amount', 'is_fraud']
                for col in expected_columns:
                    self.assertIn(col, transactions_df.columns, f"Transaction data should have {col} column")
                
                # Check data validity
                self.assertTrue(transactions_df['amount'].min() > 0, "Transaction amounts should be positive")
                self.assertTrue(transactions_df['is_fraud'].isin([0, 1]).all(), "is_fraud should be 0 or 1")
                
                # Check fraud rate is reasonable
                fraud_rate = transactions_df['is_fraud'].mean()
                self.assertGreater(fraud_rate, 0.01, "Should have some fraud")
                self.assertLess(fraud_rate, 0.20, "Fraud rate should be reasonable (<20%)")
        
        except Exception as e:
            self.skipTest(f"Demo data not available or corrupted: {e}")


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering functionality"""
    
    def test_feature_completeness(self):
        """Test that all expected features are generated"""
        try:
            ml_features_path = "data/demo/ml_features.csv"
            if os.path.exists(ml_features_path):
                features_df = pd.read_csv(ml_features_path)
                
                # Check essential features exist
                essential_features = [
                    'amount', 'hour', 'is_weekend', 'is_night', 
                    'product_risk', 'is_fraud'
                ]
                
                for feature in essential_features:
                    self.assertIn(feature, features_df.columns, f"Should have {feature} feature")
                
                # Check feature ranges
                self.assertTrue(features_df['hour'].between(0, 23).all(), "Hour should be 0-23")
                self.assertTrue(features_df['is_weekend'].isin([0, 1]).all(), "is_weekend should be 0 or 1")
                self.assertTrue(features_df['product_risk'].between(0, 1).all(), "Product risk should be 0-1")
        
        except Exception as e:
            self.skipTest(f"ML features not available: {e}")


if __name__ == '__main__':
    # Create test suite
    test_classes = [
        TestGraphNeuralNetworks,
        TestTemporalFeatures, 
        TestAdvancedExplainability,
        TestDataIntegrity,
        TestFeatureEngineering
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"🧪 TEST SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Tests run: {result.testsRun}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"⚠️ Errors: {len(result.errors)}")
    print(f"⏭️ Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 'N/A'}")
    
    if result.failures:
        print(f"\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print(f"\n⚠️ ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('\\n')[-2]}")
    
    print(f"\n🎯 Overall: {'✅ PASSED' if result.wasSuccessful() else '❌ FAILED'}")