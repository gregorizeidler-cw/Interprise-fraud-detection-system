"""
Hub Model - Central Unified Risk Model
Provides unified customer risk assessment using profile, behavioral, and network features.
"""

import logging
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import xgboost as xgb
import lightgbm as lgb
import optuna
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
from dataclasses import dataclass

from ..features.feature_store import FeatureStore
from ..data.schemas import FeatureVector, ModelPrediction
from ..infrastructure.database import DatabaseManager
from ..utils.config_manager import ConfigManager


logger = logging.getLogger(__name__)


@dataclass
class ModelTrainingConfig:
    """Configuration for model training"""
    model_name: str
    algorithm: str
    features: List[str]
    target_column: str
    validation_split: float = 0.2
    test_split: float = 0.1
    random_state: int = 42
    hyperparameter_tuning: bool = True
    cross_validation: bool = True
    cv_folds: int = 5


class BaseHubModel:
    """Base class for Hub model implementations"""
    
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.training_metrics = {}
        self.is_trained = False
    
    def prepare_training_data(self, db_manager: DatabaseManager) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data from database"""
        
        # Get labeled training data
        query = """
            SELECT 
                customer_id,
                is_fraud,
                label_timestamp,
                features_json
            FROM training_data_hub_model
            WHERE label_timestamp >= NOW() - INTERVAL '2 years'
            ORDER BY label_timestamp
        """
        
        df = db_manager.execute_query(query)
        
        if df.empty:
            raise ValueError("No training data found")
        
        # Parse features from JSON
        features_df = pd.json_normalize(df['features_json'])
        
        # Combine with labels
        X = features_df
        y = df['is_fraud'].astype(int)
        
        logger.info(f"Prepared training data: {len(X)} samples, {len(X.columns)} features")
        logger.info(f"Fraud rate: {y.mean():.4f}")
        
        return X, y
    
    def feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations"""
        
        X_engineered = X.copy()
        
        # Handle missing values
        X_engineered = X_engineered.fillna(0)
        
        # Create derived features
        if 'transaction_count_24h' in X_engineered.columns and 'transaction_volume_24h' in X_engineered.columns:
            X_engineered['avg_transaction_size_24h'] = (
                X_engineered['transaction_volume_24h'] / 
                (X_engineered['transaction_count_24h'] + 1)
            )
        
        # Risk ratios
        if 'fraudulent_beneficiaries_count' in X_engineered.columns and 'unique_beneficiaries' in X_engineered.columns:
            X_engineered['fraud_beneficiary_ratio'] = (
                X_engineered['fraudulent_beneficiaries_count'] / 
                (X_engineered['unique_beneficiaries'] + 1)
            )
        
        # Network risk indicators
        if 'network_out_degree' in X_engineered.columns and 'network_in_degree' in X_engineered.columns:
            X_engineered['network_total_degree'] = (
                X_engineered['network_out_degree'] + X_engineered['network_in_degree']
            )
        
        # Velocity features
        if 'transaction_count_1h' in X_engineered.columns and 'transaction_count_24h' in X_engineered.columns:
            X_engineered['velocity_ratio_1h_24h'] = (
                X_engineered['transaction_count_1h'] / 
                (X_engineered['transaction_count_24h'] / 24 + 0.1)
            )
        
        return X_engineered
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train the model"""
        raise NotImplementedError
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance"""
        return self.feature_importance


class XGBoostHubModel(BaseHubModel):
    """XGBoost implementation of Hub model"""
    
    def __init__(self, config: ModelTrainingConfig):
        super().__init__(config)
        self.best_params = {}
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna"""
        
        def objective(trial):
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': self.config.random_state
            }
            
            # Time series split for validation
            tscv = TimeSeriesSplit(n_splits=3)
            auc_scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_fold)
                X_val_scaled = scaler.transform(X_val_fold)
                
                model = xgb.XGBClassifier(**params)
                model.fit(X_train_scaled, y_train_fold)
                
                y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
                auc = roc_auc_score(y_val_fold, y_pred_proba)
                auc_scores.append(auc)
            
            return np.mean(auc_scores)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100, timeout=3600)
        
        self.best_params = study.best_params
        logger.info(f"Best hyperparameters: {self.best_params}")
        logger.info(f"Best AUC: {study.best_value:.4f}")
        
        return self.best_params
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train XGBoost model"""
        
        logger.info("Starting XGBoost Hub model training")
        
        # Feature engineering
        X_engineered = self.feature_engineering(X)
        
        # Split data (time series aware)
        split_point = int(len(X_engineered) * (1 - self.config.test_split))
        X_train_val, X_test = X_engineered[:split_point], X_engineered[split_point:]
        y_train_val, y_test = y[:split_point], y[split_point:]
        
        # Further split training and validation
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
        
        # Hyperparameter optimization
        if self.config.hyperparameter_tuning:
            self.optimize_hyperparameters(X_train_val, y_train_val)
        else:
            self.best_params = {
                'max_depth': 6,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'learning_rate': 0.1,
                'n_estimators': 500
            }
        
        # Train final model
        model_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': self.config.random_state,
            **self.best_params
        }
        
        self.model = xgb.XGBClassifier(**model_params)
        
        # Train with early stopping
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Evaluate model
        y_train_pred = self.model.predict_proba(X_train_scaled)[:, 1]
        y_val_pred = self.model.predict_proba(X_val_scaled)[:, 1]
        y_test_pred = self.model.predict_proba(X_test_scaled)[:, 1]
        
        self.training_metrics = {
            'train_auc': roc_auc_score(y_train, y_train_pred),
            'val_auc': roc_auc_score(y_val, y_val_pred),
            'test_auc': roc_auc_score(y_test, y_test_pred),
            'feature_count': len(X_train.columns),
            'training_samples': len(X_train)
        }
        
        # Feature importance
        self.feature_importance = dict(zip(
            X_train.columns,
            self.model.feature_importances_
        ))
        
        self.is_trained = True
        
        logger.info(f"Training completed. Test AUC: {self.training_metrics['test_auc']:.4f}")
        
        return self.training_metrics


class LightGBMHubModel(BaseHubModel):
    """LightGBM implementation of Hub model"""
    
    def __init__(self, config: ModelTrainingConfig):
        super().__init__(config)
        self.best_params = {}
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna"""
        
        def objective(trial):
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': self.config.random_state,
                'verbose': -1
            }
            
            # Time series split for validation
            tscv = TimeSeriesSplit(n_splits=3)
            auc_scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_fold)
                X_val_scaled = scaler.transform(X_val_fold)
                
                model = lgb.LGBMClassifier(**params, n_estimators=500)
                model.fit(X_train_scaled, y_train_fold)
                
                y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
                auc = roc_auc_score(y_val_fold, y_pred_proba)
                auc_scores.append(auc)
            
            return np.mean(auc_scores)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100, timeout=3600)
        
        self.best_params = study.best_params
        logger.info(f"Best hyperparameters: {self.best_params}")
        logger.info(f"Best AUC: {study.best_value:.4f}")
        
        return self.best_params
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train LightGBM model"""
        
        logger.info("Starting LightGBM Hub model training")
        
        # Feature engineering
        X_engineered = self.feature_engineering(X)
        
        # Split data (time series aware)
        split_point = int(len(X_engineered) * (1 - self.config.test_split))
        X_train_val, X_test = X_engineered[:split_point], X_engineered[split_point:]
        y_train_val, y_test = y[:split_point], y[split_point:]
        
        # Further split training and validation
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
        
        # Hyperparameter optimization
        if self.config.hyperparameter_tuning:
            self.optimize_hyperparameters(X_train_val, y_train_val)
        else:
            self.best_params = {
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 20
            }
        
        # Train final model
        model_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'random_state': self.config.random_state,
            'verbose': -1,
            'n_estimators': 1000,
            **self.best_params
        }
        
        self.model = lgb.LGBMClassifier(**model_params)
        
        # Train with early stopping
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Evaluate model
        y_train_pred = self.model.predict_proba(X_train_scaled)[:, 1]
        y_val_pred = self.model.predict_proba(X_val_scaled)[:, 1]
        y_test_pred = self.model.predict_proba(X_test_scaled)[:, 1]
        
        self.training_metrics = {
            'train_auc': roc_auc_score(y_train, y_train_pred),
            'val_auc': roc_auc_score(y_val, y_val_pred),
            'test_auc': roc_auc_score(y_test, y_test_pred),
            'feature_count': len(X_train.columns),
            'training_samples': len(X_train)
        }
        
        # Feature importance
        self.feature_importance = dict(zip(
            X_train.columns,
            self.model.feature_importances_
        ))
        
        self.is_trained = True
        
        logger.info(f"Training completed. Test AUC: {self.training_metrics['test_auc']:.4f}")
        
        return self.training_metrics


class HubModelManager:
    """
    Manager for Hub model training, evaluation, and serving.
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.db = DatabaseManager(config_manager)
        self.feature_store = FeatureStore(config_manager)
        self.model = None
        self.model_metadata = {}
    
    def create_training_dataset(self) -> bool:
        """Create training dataset for Hub model"""
        
        logger.info("Creating training dataset for Hub model")
        
        # Get customers with fraud labels
        label_query = """
            SELECT DISTINCT 
                customer_id,
                is_fraud,
                label_timestamp
            FROM fraud_labels
            WHERE label_timestamp >= NOW() - INTERVAL '2 years'
            AND confidence_score >= 0.8
        """
        
        labeled_customers = self.db.execute_query(label_query)
        
        if labeled_customers.empty:
            logger.error("No labeled customers found")
            return False
        
        training_data = []
        
        for _, row in labeled_customers.iterrows():
            customer_id = row['customer_id']
            is_fraud = row['is_fraud']
            label_timestamp = row['label_timestamp']
            
            try:
                # Get feature vector at the time of labeling
                feature_vector = self.feature_store.get_feature_vector(customer_id)
                
                # Combine all non-contextual features
                features = {}
                features.update(feature_vector.profile_features)
                features.update(feature_vector.behavioral_features)
                features.update(feature_vector.network_features)
                
                training_record = {
                    'customer_id': customer_id,
                    'is_fraud': is_fraud,
                    'label_timestamp': label_timestamp,
                    'features_json': features
                }
                
                training_data.append(training_record)
                
            except Exception as e:
                logger.error(f"Error creating training record for {customer_id}: {e}")
        
        # Save training dataset
        training_df = pd.DataFrame(training_data)
        
        # Save to database
        training_df.to_sql(
            'training_data_hub_model',
            self.db.engine,
            if_exists='replace',
            index=False
        )
        
        logger.info(f"Created training dataset with {len(training_df)} samples")
        logger.info(f"Fraud rate: {training_df['is_fraud'].mean():.4f}")
        
        return True
    
    def train_model(self, algorithm: str = "xgboost") -> bool:
        """Train Hub model"""
        
        # Create training dataset if needed
        if not self.create_training_dataset():
            return False
        
        # Configure training
        training_config = ModelTrainingConfig(
            model_name="unified_risk_hub_model",
            algorithm=algorithm,
            features=["profile", "behavioral", "network"],
            target_column="is_fraud",
            validation_split=0.2,
            test_split=0.1,
            hyperparameter_tuning=True
        )
        
        # Initialize model
        if algorithm == "xgboost":
            self.model = XGBoostHubModel(training_config)
        elif algorithm == "lightgbm":
            self.model = LightGBMHubModel(training_config)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Prepare training data
        X, y = self.model.prepare_training_data(self.db)
        
        # Train model
        with mlflow.start_run(run_name=f"hub_model_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            training_metrics = self.model.train(X, y)
            
            # Log metrics to MLflow
            mlflow.log_metrics(training_metrics)
            mlflow.log_param("algorithm", algorithm)
            mlflow.log_param("feature_count", training_metrics['feature_count'])
            mlflow.log_param("training_samples", training_metrics['training_samples'])
            
            # Log model
            if algorithm == "xgboost":
                mlflow.xgboost.log_model(self.model.model, "model")
            elif algorithm == "lightgbm":
                mlflow.lightgbm.log_model(self.model.model, "model")
            
            # Save model metadata
            self.model_metadata = {
                'model_name': training_config.model_name,
                'algorithm': algorithm,
                'version': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'training_metrics': training_metrics,
                'feature_importance': self.model.get_feature_importance(),
                'trained_at': datetime.now().isoformat()
            }
        
        # Save model to disk
        self.save_model()
        
        logger.info("Hub model training completed successfully")
        return True
    
    def save_model(self):
        """Save trained model to disk"""
        
        if not self.model or not self.model.is_trained:
            raise ValueError("No trained model to save")
        
        model_path = f"models/hub_model_{self.model_metadata['version']}.pkl"
        
        model_artifact = {
            'model': self.model.model,
            'scaler': self.model.scaler,
            'metadata': self.model_metadata,
            'feature_importance': self.model.feature_importance
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_artifact, f)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load trained model from disk"""
        
        with open(model_path, 'rb') as f:
            model_artifact = pickle.load(f)
        
        # Reconstruct model object
        algorithm = model_artifact['metadata']['algorithm']
        
        training_config = ModelTrainingConfig(
            model_name="unified_risk_hub_model",
            algorithm=algorithm,
            features=["profile", "behavioral", "network"],
            target_column="is_fraud"
        )
        
        if algorithm == "xgboost":
            self.model = XGBoostHubModel(training_config)
        elif algorithm == "lightgbm":
            self.model = LightGBMHubModel(training_config)
        
        # Restore model state
        self.model.model = model_artifact['model']
        self.model.scaler = model_artifact['scaler']
        self.model.feature_importance = model_artifact['feature_importance']
        self.model.is_trained = True
        
        self.model_metadata = model_artifact['metadata']
        
        logger.info(f"Model loaded from {model_path}")
    
    def predict_risk_score(self, customer_id: str) -> float:
        """Get unified risk score for customer"""
        
        if not self.model or not self.model.is_trained:
            raise ValueError("Model must be trained or loaded before making predictions")
        
        # Get feature vector
        feature_vector = self.feature_store.get_feature_vector(customer_id)
        
        # Combine features (excluding contextual)
        features = {}
        features.update(feature_vector.profile_features)
        features.update(feature_vector.behavioral_features)
        features.update(feature_vector.network_features)
        
        # Convert to DataFrame
        X = pd.DataFrame([features])
        
        # Apply feature engineering
        X_engineered = self.model.feature_engineering(X)
        
        # Make prediction
        risk_score = self.model.predict(X_engineered)[0]
        
        return float(risk_score)
    
    def get_model_explanation(self, customer_id: str, top_n: int = 10) -> Dict[str, Any]:
        """Get model explanation for a prediction"""
        
        risk_score = self.predict_risk_score(customer_id)
        
        # Get top feature importances
        sorted_features = sorted(
            self.model.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        top_features = dict(sorted_features[:top_n])
        
        return {
            'customer_id': customer_id,
            'risk_score': risk_score,
            'model_version': self.model_metadata.get('version'),
            'top_features': top_features,
            'prediction_timestamp': datetime.now().isoformat()
        }