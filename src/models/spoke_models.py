"""
Spoke Models - Product-Specific Fraud Detection Models
Specialized models for different financial products that use Hub model scores and contextual features.
"""

import logging
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras
import optuna
import mlflow
from dataclasses import dataclass

from ..features.feature_store import FeatureStore
from ..data.schemas import FeatureVector, ModelPrediction, ProductType
from ..infrastructure.database import DatabaseManager
from ..utils.config_manager import ConfigManager
from .hub_model import HubModelManager


logger = logging.getLogger(__name__)


@dataclass
class SpokeModelConfig:
    """Configuration for spoke model"""
    model_name: str
    product_type: ProductType
    algorithm: str
    contextual_features: List[str]
    hub_model_required: bool = True
    validation_split: float = 0.2
    test_split: float = 0.1
    random_state: int = 42


class BaseSpokeModel(ABC):
    """Base class for all spoke models"""
    
    def __init__(self, config: SpokeModelConfig, hub_model_manager: HubModelManager):
        self.config = config
        self.hub_model_manager = hub_model_manager
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.training_metrics = {}
        self.is_trained = False
    
    def prepare_training_data(self, db_manager: DatabaseManager) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data for spoke model"""
        
        # Get labeled transactions for this product type
        query = f"""
            SELECT 
                t.transaction_id,
                t.customer_id,
                t.amount,
                t.timestamp,
                t.channel,
                t.device_id,
                t.beneficiary_id,
                t.merchant_category,
                t.location_lat,
                t.location_lon,
                fl.is_fraud,
                fl.label_timestamp
            FROM transactions t
            JOIN fraud_labels fl ON t.transaction_id = fl.transaction_id
            WHERE t.product_type = '{self.config.product_type.value}'
            AND fl.label_timestamp >= NOW() - INTERVAL '2 years'
            AND fl.confidence_score >= 0.8
            ORDER BY t.timestamp
        """
        
        df = db_manager.execute_query(query)
        
        if df.empty:
            raise ValueError(f"No training data found for {self.config.product_type}")
        
        # Get hub model scores for each transaction
        hub_scores = []
        contextual_features_list = []
        
        for _, row in df.iterrows():
            try:
                # Get hub model risk score
                if self.config.hub_model_required:
                    hub_score = self.hub_model_manager.predict_risk_score(row['customer_id'])
                else:
                    hub_score = 0.0
                
                hub_scores.append(hub_score)
                
                # Get contextual features for this transaction
                transaction_context = {
                    'transaction': {
                        'amount': row['amount'],
                        'timestamp': row['timestamp'].isoformat(),
                        'channel': row['channel'],
                        'device_id': row['device_id'],
                        'beneficiary_id': row['beneficiary_id'],
                        'merchant_category': row['merchant_category'],
                        'location_lat': row['location_lat'],
                        'location_lon': row['location_lon']
                    }
                }
                
                contextual_features = self._extract_contextual_features(
                    row['customer_id'], transaction_context
                )
                contextual_features_list.append(contextual_features)
                
            except Exception as e:
                logger.error(f"Error processing transaction {row['transaction_id']}: {e}")
                hub_scores.append(0.0)
                contextual_features_list.append({})
        
        # Create feature matrix
        features_df = pd.DataFrame(contextual_features_list)
        features_df['hub_risk_score'] = hub_scores
        
        # Fill missing values
        features_df = features_df.fillna(0)
        
        # Target variable
        y = df['is_fraud'].astype(int)
        
        logger.info(f"Prepared {self.config.product_type} training data: {len(features_df)} samples")
        logger.info(f"Fraud rate: {y.mean():.4f}")
        
        return features_df, y
    
    def _extract_contextual_features(self, customer_id: str, transaction_context: Dict) -> Dict[str, Any]:
        """Extract contextual features for transaction"""
        
        feature_store = FeatureStore(ConfigManager())
        feature_vector = feature_store.get_feature_vector(customer_id, transaction_context)
        
        return feature_vector.contextual_features
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train the spoke model"""
        pass
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def predict_with_explanation(self, customer_id: str, transaction_context: Dict) -> Dict[str, Any]:
        """Make prediction with explanation"""
        
        # Get hub score
        if self.config.hub_model_required:
            hub_score = self.hub_model_manager.predict_risk_score(customer_id)
        else:
            hub_score = 0.0
        
        # Get contextual features
        contextual_features = self._extract_contextual_features(customer_id, transaction_context)
        contextual_features['hub_risk_score'] = hub_score
        
        # Create feature vector
        X = pd.DataFrame([contextual_features])
        X = X.fillna(0)
        
        # Make prediction
        fraud_probability = self.predict(X)[0]
        
        return {
            'customer_id': customer_id,
            'product_type': self.config.product_type.value,
            'fraud_probability': float(fraud_probability),
            'hub_risk_score': hub_score,
            'contextual_features': contextual_features,
            'model_name': self.config.model_name,
            'prediction_timestamp': datetime.now().isoformat()
        }


class PIXSpokeModel(BaseSpokeModel):
    """PIX (Instant Payment) specific fraud detection model"""
    
    def __init__(self, hub_model_manager: HubModelManager):
        config = SpokeModelConfig(
            model_name="pix_fraud_model",
            product_type=ProductType.PIX,
            algorithm="lightgbm",
            contextual_features=[
                "transaction_amount",
                "is_round_amount", 
                "is_weekend",
                "hour_of_day",
                "is_night_transaction",
                "minutes_since_last_transaction",
                "distance_from_usual_location_km",
                "amount_zscore",
                "is_new_beneficiary"
            ]
        )
        super().__init__(config, hub_model_manager)
    
    def _extract_contextual_features(self, customer_id: str, transaction_context: Dict) -> Dict[str, Any]:
        """Extract PIX-specific contextual features"""
        
        base_features = super()._extract_contextual_features(customer_id, transaction_context)
        
        # Add PIX-specific features
        transaction = transaction_context.get('transaction', {})
        beneficiary_id = transaction.get('beneficiary_id')
        
        # Check if beneficiary is new
        if beneficiary_id:
            db = DatabaseManager(ConfigManager())
            query = """
                SELECT COUNT(*) as previous_transfers
                FROM transactions 
                WHERE customer_id = %s 
                AND beneficiary_id = %s
                AND timestamp < %s
                AND product_type = 'pix'
            """
            
            result = db.execute_query(query, params=[
                customer_id, beneficiary_id, transaction.get('timestamp')
            ])
            
            is_new_beneficiary = int(result.iloc[0]['previous_transfers'] == 0)
        else:
            is_new_beneficiary = 1
        
        base_features['is_new_beneficiary'] = is_new_beneficiary
        
        return base_features
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train PIX fraud detection model"""
        
        logger.info("Training PIX spoke model")
        
        # Split data
        split_point = int(len(X) * (1 - self.config.test_split))
        X_train_val, X_test = X[:split_point], X[split_point:]
        y_train_val, y_test = y[:split_point], y[split_point:]
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=self.config.validation_split,
            random_state=self.config.random_state,
            stratify=y_train_val
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train LightGBM model
        self.model = lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            num_leaves=31,
            learning_rate=0.1,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbose=-1,
            random_state=self.config.random_state,
            n_estimators=500
        )
        
        # Train with early stopping
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Evaluate
        y_test_pred = self.model.predict_proba(X_test_scaled)[:, 1]
        test_auc = roc_auc_score(y_test, y_test_pred)
        
        self.training_metrics = {
            'test_auc': test_auc,
            'feature_count': len(X_train.columns),
            'training_samples': len(X_train)
        }
        
        # Feature importance
        self.feature_importance = dict(zip(X_train.columns, self.model.feature_importances_))
        self.is_trained = True
        
        logger.info(f"PIX model training completed. Test AUC: {test_auc:.4f}")
        
        return self.training_metrics


class CreditCardSpokeModel(BaseSpokeModel):
    """Credit Card specific fraud detection model"""
    
    def __init__(self, hub_model_manager: HubModelManager):
        config = SpokeModelConfig(
            model_name="credit_card_fraud_model",
            product_type=ProductType.CREDIT_CARD,
            algorithm="xgboost",
            contextual_features=[
                "transaction_amount",
                "merchant_category_encoded",
                "is_online_transaction",
                "distance_from_usual_location_km",
                "amount_zscore",
                "merchant_risk_score",
                "card_present"
            ]
        )
        super().__init__(config, hub_model_manager)
    
    def _extract_contextual_features(self, customer_id: str, transaction_context: Dict) -> Dict[str, Any]:
        """Extract credit card specific contextual features"""
        
        base_features = super()._extract_contextual_features(customer_id, transaction_context)
        
        transaction = transaction_context.get('transaction', {})
        merchant_category = transaction.get('merchant_category')
        
        # Encode merchant category
        high_risk_categories = ['5812', '5813', '7995', '5933']  # Restaurants, bars, gambling, jewelry
        merchant_risk = 1 if merchant_category in high_risk_categories else 0
        
        # Category encoding (simplified)
        category_encoding = {
            '5411': 1,  # Grocery
            '5541': 2,  # Gas stations
            '5812': 3,  # Restaurants
            '5999': 4,  # Misc retail
        }
        
        base_features.update({
            'merchant_category_encoded': category_encoding.get(merchant_category, 0),
            'is_online_transaction': int(transaction.get('channel') == 'web_browser'),
            'merchant_risk_score': merchant_risk,
            'card_present': int(transaction.get('channel') in ['pos', 'atm'])
        })
        
        return base_features
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train credit card fraud detection model"""
        
        logger.info("Training Credit Card spoke model")
        
        # Split data
        split_point = int(len(X) * (1 - self.config.test_split))
        X_train_val, X_test = X[:split_point], X[split_point:]
        y_train_val, y_test = y[:split_point], y[split_point:]
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=self.config.validation_split,
            random_state=self.config.random_state,
            stratify=y_train_val
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost model
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            learning_rate=0.1,
            n_estimators=500,
            random_state=self.config.random_state
        )
        
        # Train with early stopping
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Evaluate
        y_test_pred = self.model.predict_proba(X_test_scaled)[:, 1]
        test_auc = roc_auc_score(y_test, y_test_pred)
        
        self.training_metrics = {
            'test_auc': test_auc,
            'feature_count': len(X_train.columns),
            'training_samples': len(X_train)
        }
        
        # Feature importance
        self.feature_importance = dict(zip(X_train.columns, self.model.feature_importances_))
        self.is_trained = True
        
        logger.info(f"Credit Card model training completed. Test AUC: {test_auc:.4f}")
        
        return self.training_metrics


class LoanSpokeModel(BaseSpokeModel):
    """Loan application fraud detection model using neural networks"""
    
    def __init__(self, hub_model_manager: HubModelManager):
        config = SpokeModelConfig(
            model_name="loan_fraud_model",
            product_type=ProductType.LOAN,
            algorithm="tensorflow",
            contextual_features=[
                "loan_amount",
                "loan_purpose_encoded",
                "debt_to_income_ratio",
                "employment_stability_score",
                "collateral_value_ratio",
                "interest_rate"
            ]
        )
        super().__init__(config, hub_model_manager)
    
    def _extract_contextual_features(self, customer_id: str, transaction_context: Dict) -> Dict[str, Any]:
        """Extract loan-specific contextual features"""
        
        # For loans, we'd extract from loan application data
        # This is a simplified implementation
        
        transaction = transaction_context.get('transaction', {})
        
        # Mock loan features (in reality, these would come from loan application)
        loan_features = {
            'loan_amount': float(transaction.get('amount', 0)),
            'loan_purpose_encoded': 1,  # Would be encoded from purpose
            'debt_to_income_ratio': 0.3,  # Would be calculated
            'employment_stability_score': 0.8,  # Would be calculated
            'collateral_value_ratio': 1.2,  # Would be calculated
            'interest_rate': 0.15  # Would come from loan terms
        }
        
        return loan_features
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train loan fraud detection model using neural network"""
        
        logger.info("Training Loan spoke model with neural network")
        
        # Split data
        split_point = int(len(X) * (1 - self.config.test_split))
        X_train_val, X_test = X[:split_point], X[split_point:]
        y_train_val, y_test = y[:split_point], y[split_point:]
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=self.config.validation_split,
            random_state=self.config.random_state,
            stratify=y_train_val
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Build neural network
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['auc']
        )
        
        # Train model
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=10,
            restore_best_weights=True
        )
        
        history = model.fit(
            X_train_scaled, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val_scaled, y_val),
            callbacks=[early_stopping],
            verbose=0
        )
        
        self.model = model
        
        # Evaluate
        y_test_pred = self.model.predict(X_test_scaled).flatten()
        test_auc = roc_auc_score(y_test, y_test_pred)
        
        self.training_metrics = {
            'test_auc': test_auc,
            'feature_count': len(X_train.columns),
            'training_samples': len(X_train),
            'epochs_trained': len(history.history['loss'])
        }
        
        self.is_trained = True
        
        logger.info(f"Loan model training completed. Test AUC: {test_auc:.4f}")
        
        return self.training_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with neural network"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled).flatten()


class SpokeModelManager:
    """
    Manager for all spoke models - orchestrates training and serving.
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.db = DatabaseManager(config_manager)
        self.hub_model_manager = HubModelManager(config_manager)
        self.spoke_models = {}
        
        # Initialize spoke models
        self.spoke_models['pix'] = PIXSpokeModel(self.hub_model_manager)
        self.spoke_models['credit_card'] = CreditCardSpokeModel(self.hub_model_manager)
        self.spoke_models['loan'] = LoanSpokeModel(self.hub_model_manager)
    
    def train_all_models(self) -> Dict[str, Dict[str, Any]]:
        """Train all spoke models"""
        
        results = {}
        
        for product_type, model in self.spoke_models.items():
            try:
                logger.info(f"Training {product_type} spoke model")
                
                # Prepare training data
                X, y = model.prepare_training_data(self.db)
                
                # Train model
                metrics = model.train(X, y)
                results[product_type] = metrics
                
                # Save model
                self.save_spoke_model(product_type, model)
                
            except Exception as e:
                logger.error(f"Failed to train {product_type} model: {e}")
                results[product_type] = {'error': str(e)}
        
        return results
    
    def train_specific_model(self, product_type: str) -> Dict[str, Any]:
        """Train a specific spoke model"""
        
        if product_type not in self.spoke_models:
            raise ValueError(f"Unknown product type: {product_type}")
        
        model = self.spoke_models[product_type]
        
        # Prepare training data
        X, y = model.prepare_training_data(self.db)
        
        # Train model
        metrics = model.train(X, y)
        
        # Save model
        self.save_spoke_model(product_type, model)
        
        return metrics
    
    def save_spoke_model(self, product_type: str, model: BaseSpokeModel):
        """Save spoke model to disk"""
        
        if not model.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_path = f"models/{product_type}_spoke_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        model_artifact = {
            'model': model.model,
            'scaler': model.scaler,
            'config': model.config,
            'training_metrics': model.training_metrics,
            'feature_importance': model.feature_importance
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_artifact, f)
        
        logger.info(f"{product_type} model saved to {model_path}")
    
    def load_spoke_model(self, product_type: str, model_path: str):
        """Load spoke model from disk"""
        
        with open(model_path, 'rb') as f:
            model_artifact = pickle.load(f)
        
        config = model_artifact['config']
        
        # Recreate model object
        if product_type == 'pix':
            model = PIXSpokeModel(self.hub_model_manager)
        elif product_type == 'credit_card':
            model = CreditCardSpokeModel(self.hub_model_manager)
        elif product_type == 'loan':
            model = LoanSpokeModel(self.hub_model_manager)
        else:
            raise ValueError(f"Unknown product type: {product_type}")
        
        # Restore model state
        model.model = model_artifact['model']
        model.scaler = model_artifact['scaler']
        model.training_metrics = model_artifact['training_metrics']
        model.feature_importance = model_artifact['feature_importance']
        model.is_trained = True
        
        self.spoke_models[product_type] = model
        
        logger.info(f"{product_type} model loaded from {model_path}")
    
    def predict(self, product_type: str, customer_id: str, transaction_context: Dict) -> Dict[str, Any]:
        """Make prediction using appropriate spoke model"""
        
        if product_type not in self.spoke_models:
            raise ValueError(f"Unknown product type: {product_type}")
        
        model = self.spoke_models[product_type]
        
        if not model.is_trained:
            raise ValueError(f"{product_type} model is not trained")
        
        return model.predict_with_explanation(customer_id, transaction_context)
    
    def get_model_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for all spoke models"""
        
        performance = {}
        
        for product_type, model in self.spoke_models.items():
            if model.is_trained:
                performance[product_type] = {
                    'training_metrics': model.training_metrics,
                    'feature_importance': dict(
                        sorted(model.feature_importance.items(), 
                              key=lambda x: x[1], reverse=True)[:10]
                    )
                }
            else:
                performance[product_type] = {'status': 'not_trained'}
        
        return performance