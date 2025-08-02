"""
Advanced Explainability for Fraud Detection Models

This module implements state-of-the-art explainability techniques for fraud detection,
including SHAP (SHapley Additive exPlanations), counterfactual explanations,
and advanced interpretability methods for compliance and model transparency.

Key Features:
- SHAP explanations (TreeExplainer, KernelExplainer, DeepExplainer)
- Counterfactual generation using DiCE framework
- LIME explanations for local interpretability
- Feature importance analysis and ranking
- Global and local explanation generation
- Model-agnostic explanation methods
- Compliance reporting and documentation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from datetime import datetime
import json
import pickle

# SHAP imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

# LIME imports
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("LIME not available. Install with: pip install lime")

# Counterfactual imports
try:
    import dice_ml
    DICE_AVAILABLE = True
except ImportError:
    DICE_AVAILABLE = False
    print("DiCE not available. Install with: pip install dice-ml")

# Scientific computing
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    Advanced SHAP-based explanations for fraud detection models
    """
    
    def __init__(self, model, model_type: str = 'auto'):
        """
        Initialize SHAP explainer
        
        Args:
            model: Trained model to explain
            model_type: Type of model ('tree', 'linear', 'deep', 'kernel', 'auto')
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required. Install with: pip install shap")
        
        self.model = model
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
        self.background_data = None
        
    def initialize_explainer(self, 
                           X_train: pd.DataFrame, 
                           max_background_samples: int = 100):
        """
        Initialize the appropriate SHAP explainer based on model type
        
        Args:
            X_train: Training data for background distribution
            max_background_samples: Maximum samples for background data
        """
        self.feature_names = X_train.columns.tolist()
        
        # Sample background data for efficiency
        if len(X_train) > max_background_samples:
            background_indices = np.random.choice(
                len(X_train), max_background_samples, replace=False
            )
            self.background_data = X_train.iloc[background_indices]
        else:
            self.background_data = X_train
        
        # Auto-detect model type if not specified
        if self.model_type == 'auto':
            self.model_type = self._detect_model_type()
        
        # Initialize appropriate explainer
        if self.model_type == 'tree':
            self.explainer = shap.TreeExplainer(
                self.model, 
                data=self.background_data,
                feature_perturbation='interventional'
            )
        elif self.model_type == 'linear':
            self.explainer = shap.LinearExplainer(
                self.model, 
                self.background_data
            )
        elif self.model_type == 'deep':
            self.explainer = shap.DeepExplainer(
                self.model, 
                self.background_data.values
            )
        else:  # kernel explainer as fallback
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, 
                self.background_data,
                link="logit"
            )
        
        logger.info(f"Initialized {self.model_type} SHAP explainer")
    
    def _detect_model_type(self) -> str:
        """Auto-detect model type for appropriate explainer"""
        model_class = self.model.__class__.__name__
        
        if any(tree_type in model_class.lower() for tree_type in 
               ['tree', 'forest', 'boosting', 'xgb', 'lgb', 'catboost']):
            return 'tree'
        elif any(linear_type in model_class.lower() for linear_type in 
                ['linear', 'logistic', 'ridge', 'lasso']):
            return 'linear'
        elif any(deep_type in model_class.lower() for deep_type in 
                ['neural', 'mlp', 'deep', 'keras', 'torch']):
            return 'deep'
        else:
            return 'kernel'
    
    def explain_predictions(self, 
                          X_test: pd.DataFrame, 
                          max_samples: int = None) -> np.ndarray:
        """
        Generate SHAP explanations for test data
        
        Args:
            X_test: Test data to explain
            max_samples: Maximum number of samples to explain
            
        Returns:
            SHAP values array
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer first.")
        
        if max_samples and len(X_test) > max_samples:
            sample_indices = np.random.choice(len(X_test), max_samples, replace=False)
            X_explain = X_test.iloc[sample_indices]
        else:
            X_explain = X_test
        
        logger.info(f"Generating SHAP explanations for {len(X_explain)} samples...")
        
        # Generate SHAP values
        if self.model_type == 'deep':
            self.shap_values = self.explainer.shap_values(X_explain.values)
        else:
            self.shap_values = self.explainer.shap_values(X_explain)
        
        # For binary classification, get positive class SHAP values
        if isinstance(self.shap_values, list) and len(self.shap_values) == 2:
            self.shap_values = self.shap_values[1]  # Positive class (fraud)
        
        logger.info("SHAP explanations generated successfully")
        return self.shap_values
    
    def get_feature_importance(self, global_importance: bool = True) -> pd.DataFrame:
        """
        Get feature importance based on SHAP values
        
        Args:
            global_importance: Whether to compute global (True) or local (False) importance
            
        Returns:
            DataFrame with feature importance scores
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call explain_predictions first.")
        
        if global_importance:
            # Global importance: mean absolute SHAP value
            importance_scores = np.mean(np.abs(self.shap_values), axis=0)
        else:
            # Local importance: SHAP values for individual predictions
            importance_scores = self.shap_values
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores if global_importance else importance_scores.mean(axis=0),
            'abs_importance': np.abs(importance_scores) if global_importance else np.abs(importance_scores).mean(axis=0)
        })
        
        return importance_df.sort_values('abs_importance', ascending=False)
    
    def plot_summary(self, 
                    X_test: pd.DataFrame = None, 
                    plot_type: str = 'dot',
                    max_display: int = 20,
                    save_path: str = None) -> None:
        """
        Create SHAP summary plot
        
        Args:
            X_test: Test data for feature values
            plot_type: Type of plot ('dot', 'bar', 'violin')
            max_display: Maximum features to display
            save_path: Path to save the plot
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call explain_predictions first.")
        
        plt.figure(figsize=(12, 8))
        
        if plot_type == 'bar':
            shap.summary_plot(
                self.shap_values, 
                feature_names=self.feature_names,
                plot_type='bar',
                max_display=max_display,
                show=False
            )
        else:
            # For dot and violin plots, we need feature values
            if X_test is not None:
                feature_values = X_test.iloc[:len(self.shap_values)]
            else:
                feature_values = None
            
            shap.summary_plot(
                self.shap_values, 
                features=feature_values,
                feature_names=self.feature_names,
                plot_type=plot_type,
                max_display=max_display,
                show=False
            )
        
        plt.title('SHAP Feature Importance Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"SHAP summary plot saved to {save_path}")
        
        plt.show()
    
    def plot_waterfall(self, 
                      sample_idx: int, 
                      X_test: pd.DataFrame = None,
                      save_path: str = None) -> None:
        """
        Create SHAP waterfall plot for a single prediction
        
        Args:
            sample_idx: Index of sample to explain
            X_test: Test data for feature values
            save_path: Path to save the plot
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call explain_predictions first.")
        
        if sample_idx >= len(self.shap_values):
            raise ValueError(f"Sample index {sample_idx} out of range")
        
        # Create explanation object
        if X_test is not None:
            feature_values = X_test.iloc[sample_idx]
        else:
            feature_values = None
        
        # Get expected value (baseline)
        if hasattr(self.explainer, 'expected_value'):
            expected_value = self.explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)):
                expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
        else:
            expected_value = 0
        
        # Create explanation object
        explanation = shap.Explanation(
            values=self.shap_values[sample_idx],
            base_values=expected_value,
            data=feature_values.values if feature_values is not None else None,
            feature_names=self.feature_names
        )
        
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(explanation, show=False)
        plt.title(f'SHAP Waterfall Plot - Sample {sample_idx}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"SHAP waterfall plot saved to {save_path}")
        
        plt.show()
    
    def get_explanation_text(self, 
                           sample_idx: int, 
                           X_test: pd.DataFrame,
                           top_features: int = 5) -> str:
        """
        Generate human-readable explanation text
        
        Args:
            sample_idx: Index of sample to explain
            X_test: Test data
            top_features: Number of top features to include
            
        Returns:
            Human-readable explanation string
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call explain_predictions first.")
        
        sample_shap = self.shap_values[sample_idx]
        sample_features = X_test.iloc[sample_idx]
        
        # Get top positive and negative contributors
        feature_contributions = list(zip(self.feature_names, sample_shap, sample_features))
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Build explanation text
        explanation_parts = []
        explanation_parts.append("🔍 **Fraud Detection Explanation**\n")
        
        # Prediction confidence
        total_shap = np.sum(sample_shap)
        if hasattr(self.explainer, 'expected_value'):
            baseline = self.explainer.expected_value
            if isinstance(baseline, (list, np.ndarray)):
                baseline = baseline[1] if len(baseline) > 1 else baseline[0]
        else:
            baseline = 0
        
        prediction_score = baseline + total_shap
        risk_level = "HIGH" if prediction_score > 0.5 else "MEDIUM" if prediction_score > 0.2 else "LOW"
        
        explanation_parts.append(f"**Risk Level: {risk_level}** (Score: {prediction_score:.3f})\n")
        
        # Top contributing features
        explanation_parts.append("**Key Factors:**")
        
        for i, (feature, shap_val, feature_val) in enumerate(feature_contributions[:top_features]):
            direction = "increases" if shap_val > 0 else "decreases"
            impact = "strongly" if abs(shap_val) > 0.1 else "moderately" if abs(shap_val) > 0.05 else "slightly"
            
            explanation_parts.append(
                f"{i+1}. **{feature}** = {feature_val:.3f} {impact} {direction} fraud risk (impact: {shap_val:+.3f})"
            )
        
        return "\n".join(explanation_parts)


class CounterfactualExplainer:
    """
    Counterfactual explanations using DiCE (Diverse Counterfactual Explanations)
    """
    
    def __init__(self, model, model_type: str = 'sklearn'):
        """
        Initialize counterfactual explainer
        
        Args:
            model: Trained model
            model_type: Type of model ('sklearn', 'tensorflow', 'pytorch')
        """
        if not DICE_AVAILABLE:
            raise ImportError("DiCE is required. Install with: pip install dice-ml")
        
        self.model = model
        self.model_type = model_type
        self.dice_model = None
        self.dice_explainer = None
        
    def initialize_explainer(self, 
                           X_train: pd.DataFrame, 
                           y_train: pd.Series,
                           continuous_features: List[str] = None,
                           categorical_features: List[str] = None):
        """
        Initialize DiCE explainer
        
        Args:
            X_train: Training features
            y_train: Training targets
            continuous_features: List of continuous feature names
            categorical_features: List of categorical feature names
        """
        # Prepare training data
        train_data = X_train.copy()
        train_data['target'] = y_train
        
        # Auto-detect feature types if not provided
        if continuous_features is None:
            continuous_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        if categorical_features is None:
            categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Create DiCE data object
        self.dice_data = dice_ml.Data(
            dataframe=train_data,
            continuous_features=continuous_features,
            outcome_name='target'
        )
        
        # Create DiCE model object
        if self.model_type == 'sklearn':
            self.dice_model = dice_ml.Model(
                model=self.model,
                backend='sklearn'
            )
        else:
            raise NotImplementedError(f"Model type {self.model_type} not implemented")
        
        # Create explainer
        self.dice_explainer = dice_ml.Dice(
            self.dice_data,
            self.dice_model,
            method='random'
        )
        
        logger.info("DiCE counterfactual explainer initialized")
    
    def generate_counterfactuals(self, 
                               query_instance: pd.DataFrame,
                               num_counterfactuals: int = 5,
                               desired_class: int = 0,
                               proximity_weight: float = 0.5,
                               diversity_weight: float = 1.0) -> Dict:
        """
        Generate counterfactual explanations
        
        Args:
            query_instance: Instance to generate counterfactuals for
            num_counterfactuals: Number of counterfactuals to generate
            desired_class: Desired outcome class (0 for non-fraud, 1 for fraud)
            proximity_weight: Weight for proximity constraint
            diversity_weight: Weight for diversity constraint
            
        Returns:
            Dictionary containing counterfactuals and analysis
        """
        if self.dice_explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer first.")
        
        # Generate counterfactuals
        counterfactuals = self.dice_explainer.generate_counterfactuals(
            query_instance,
            total_CFs=num_counterfactuals,
            desired_class=desired_class,
            proximity_weight=proximity_weight,
            diversity_weight=diversity_weight
        )
        
        # Extract counterfactual data
        cf_examples = counterfactuals.cf_examples_list[0]
        original_instance = query_instance.iloc[0]
        
        # Analyze counterfactuals
        analysis = self._analyze_counterfactuals(original_instance, cf_examples)
        
        return {
            'original_instance': original_instance,
            'counterfactuals': cf_examples,
            'analysis': analysis,
            'counterfactual_object': counterfactuals
        }
    
    def _analyze_counterfactuals(self, 
                               original: pd.Series, 
                               counterfactuals: pd.DataFrame) -> Dict:
        """
        Analyze counterfactual examples to extract insights
        
        Args:
            original: Original instance
            counterfactuals: Generated counterfactual examples
            
        Returns:
            Analysis dictionary
        """
        analysis = {
            'num_counterfactuals': len(counterfactuals),
            'changed_features': {},
            'feature_changes': {},
            'average_changes': {},
            'actionable_insights': []
        }
        
        # Analyze feature changes
        for feature in original.index:
            if feature in counterfactuals.columns:
                original_value = original[feature]
                cf_values = counterfactuals[feature]
                
                # Find features that changed
                changes = cf_values != original_value
                if changes.any():
                    analysis['changed_features'][feature] = {
                        'change_frequency': changes.sum() / len(counterfactuals),
                        'original_value': original_value,
                        'counterfactual_values': cf_values[changes].tolist(),
                        'average_change': (cf_values - original_value).mean(),
                        'min_change': (cf_values - original_value).min(),
                        'max_change': (cf_values - original_value).max()
                    }
        
        # Generate actionable insights
        analysis['actionable_insights'] = self._generate_actionable_insights(
            analysis['changed_features']
        )
        
        return analysis
    
    def _generate_actionable_insights(self, changed_features: Dict) -> List[str]:
        """
        Generate human-readable actionable insights from counterfactuals
        """
        insights = []
        
        # Sort features by change frequency
        sorted_features = sorted(
            changed_features.items(),
            key=lambda x: x[1]['change_frequency'],
            reverse=True
        )
        
        for feature, changes in sorted_features[:5]:  # Top 5 most changed features
            freq = changes['change_frequency']
            original = changes['original_value']
            avg_change = changes['average_change']
            
            if freq > 0.5:  # Feature changed in majority of counterfactuals
                if avg_change > 0:
                    direction = "increase"
                    action = f"reducing {feature} from {original:.3f}"
                else:
                    direction = "decrease"
                    action = f"increasing {feature} from {original:.3f}"
                
                insights.append(
                    f"To change the prediction, consider {action} "
                    f"(changed in {freq*100:.1f}% of counterfactuals)"
                )
        
        return insights
    
    def plot_counterfactuals(self, 
                           counterfactual_result: Dict,
                           features_to_plot: List[str] = None,
                           save_path: str = None) -> None:
        """
        Visualize counterfactual explanations
        
        Args:
            counterfactual_result: Result from generate_counterfactuals
            features_to_plot: Specific features to plot
            save_path: Path to save the plot
        """
        original = counterfactual_result['original_instance']
        counterfactuals = counterfactual_result['counterfactuals']
        
        if features_to_plot is None:
            # Select top changed features
            changed_features = counterfactual_result['analysis']['changed_features']
            features_to_plot = list(changed_features.keys())[:6]  # Top 6 features
        
        # Create subplots
        n_features = len(features_to_plot)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, feature in enumerate(features_to_plot):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Plot original value
            ax.axvline(original[feature], color='red', linestyle='--', 
                      label=f'Original: {original[feature]:.3f}', linewidth=2)
            
            # Plot counterfactual values
            cf_values = counterfactuals[feature]
            ax.hist(cf_values, bins=10, alpha=0.7, color='blue', 
                   label=f'Counterfactuals (n={len(cf_values)})')
            
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Feature: {feature}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Counterfactual Analysis - Feature Changes', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Counterfactual plot saved to {save_path}")
        
        plt.show()


class LIMEExplainer:
    """
    LIME (Local Interpretable Model-agnostic Explanations) for fraud detection
    """
    
    def __init__(self, model, training_data: pd.DataFrame):
        """
        Initialize LIME explainer
        
        Args:
            model: Trained model to explain
            training_data: Training data for LIME explainer
        """
        if not LIME_AVAILABLE:
            raise ImportError("LIME is required. Install with: pip install lime")
        
        self.model = model
        self.training_data = training_data
        
        # Initialize LIME explainer
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=training_data.values,
            feature_names=training_data.columns.tolist(),
            class_names=['Not Fraud', 'Fraud'],
            mode='classification',
            discretize_continuous=True
        )
        
        logger.info("LIME explainer initialized")
    
    def explain_instance(self, 
                        instance: pd.Series,
                        num_features: int = 10,
                        num_samples: int = 5000) -> Dict:
        """
        Explain a single instance using LIME
        
        Args:
            instance: Instance to explain
            num_features: Number of features to include in explanation
            num_samples: Number of samples for LIME
            
        Returns:
            LIME explanation results
        """
        # Generate explanation
        explanation = self.explainer.explain_instance(
            data_row=instance.values,
            predict_fn=self.model.predict_proba,
            num_features=num_features,
            num_samples=num_samples
        )
        
        # Extract explanation data
        explanation_data = {
            'local_exp': explanation.local_exp[1],  # For fraud class
            'intercept': explanation.intercept[1],
            'predicted_proba': explanation.predict_proba[1],
            'score': explanation.score,
            'feature_names': [self.training_data.columns[i] for i, _ in explanation.local_exp[1]]
        }
        
        return {
            'explanation_object': explanation,
            'explanation_data': explanation_data,
            'instance': instance
        }
    
    def plot_explanation(self, 
                        explanation_result: Dict,
                        save_path: str = None) -> None:
        """
        Plot LIME explanation
        
        Args:
            explanation_result: Result from explain_instance
            save_path: Path to save the plot
        """
        explanation = explanation_result['explanation_object']
        
        # Create plot
        fig = explanation.as_pyplot_figure()
        fig.suptitle('LIME Local Explanation', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"LIME explanation plot saved to {save_path}")
        
        plt.show()


class ExplainabilityManager:
    """
    Main manager for all explainability techniques
    """
    
    def __init__(self, model, config: Dict = None):
        """
        Initialize explainability manager
        
        Args:
            model: Trained model to explain
            config: Configuration dictionary
        """
        self.model = model
        self.config = config or {}
        self.shap_explainer = None
        self.counterfactual_explainer = None
        self.lime_explainer = None
        self.explanations_cache = {}
        
    def initialize_all_explainers(self, 
                                X_train: pd.DataFrame, 
                                y_train: pd.Series) -> None:
        """
        Initialize all available explainers
        
        Args:
            X_train: Training features
            y_train: Training targets
        """
        logger.info("Initializing all explainability techniques...")
        
        # Initialize SHAP
        if SHAP_AVAILABLE:
            self.shap_explainer = SHAPExplainer(self.model)
            self.shap_explainer.initialize_explainer(X_train)
            logger.info("✅ SHAP explainer initialized")
        
        # Initialize Counterfactuals
        if DICE_AVAILABLE:
            self.counterfactual_explainer = CounterfactualExplainer(self.model)
            self.counterfactual_explainer.initialize_explainer(X_train, y_train)
            logger.info("✅ Counterfactual explainer initialized")
        
        # Initialize LIME
        if LIME_AVAILABLE:
            self.lime_explainer = LIMEExplainer(self.model, X_train)
            logger.info("✅ LIME explainer initialized")
        
        logger.info("All explainers initialized successfully!")
    
    def get_comprehensive_explanation(self, 
                                    instance: pd.Series,
                                    X_test: pd.DataFrame = None,
                                    include_counterfactuals: bool = True,
                                    include_lime: bool = True) -> Dict:
        """
        Get comprehensive explanation using all available methods
        
        Args:
            instance: Instance to explain
            X_test: Test dataset for SHAP context
            include_counterfactuals: Whether to include counterfactual explanations
            include_lime: Whether to include LIME explanations
            
        Returns:
            Comprehensive explanation dictionary
        """
        explanation_id = f"explanation_{hash(str(instance.values))}"
        
        if explanation_id in self.explanations_cache:
            return self.explanations_cache[explanation_id]
        
        comprehensive_explanation = {
            'instance': instance,
            'timestamp': datetime.now().isoformat(),
            'model_prediction': None,
            'shap_explanation': None,
            'counterfactual_explanation': None,
            'lime_explanation': None,
            'summary': None
        }
        
        # Get model prediction
        if hasattr(self.model, 'predict_proba'):
            pred_proba = self.model.predict_proba([instance.values])[0]
            comprehensive_explanation['model_prediction'] = {
                'fraud_probability': pred_proba[1],
                'prediction': int(pred_proba[1] > 0.5)
            }
        
        # SHAP explanation
        if self.shap_explainer:
            try:
                instance_df = pd.DataFrame([instance])
                shap_values = self.shap_explainer.explain_predictions(instance_df)
                
                comprehensive_explanation['shap_explanation'] = {
                    'shap_values': shap_values[0].tolist(),
                    'feature_names': self.shap_explainer.feature_names,
                    'explanation_text': self.shap_explainer.get_explanation_text(0, instance_df)
                }
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}")
        
        # Counterfactual explanation
        if self.counterfactual_explainer and include_counterfactuals:
            try:
                instance_df = pd.DataFrame([instance])
                cf_result = self.counterfactual_explainer.generate_counterfactuals(
                    instance_df,
                    num_counterfactuals=3,
                    desired_class=0  # Non-fraud
                )
                
                comprehensive_explanation['counterfactual_explanation'] = {
                    'counterfactuals': cf_result['counterfactuals'].to_dict('records'),
                    'actionable_insights': cf_result['analysis']['actionable_insights'],
                    'changed_features': cf_result['analysis']['changed_features']
                }
            except Exception as e:
                logger.warning(f"Counterfactual explanation failed: {e}")
        
        # LIME explanation
        if self.lime_explainer and include_lime:
            try:
                lime_result = self.lime_explainer.explain_instance(instance)
                
                comprehensive_explanation['lime_explanation'] = {
                    'feature_weights': lime_result['explanation_data']['local_exp'],
                    'intercept': lime_result['explanation_data']['intercept'],
                    'predicted_proba': lime_result['explanation_data']['predicted_proba']
                }
            except Exception as e:
                logger.warning(f"LIME explanation failed: {e}")
        
        # Generate summary
        comprehensive_explanation['summary'] = self._generate_explanation_summary(
            comprehensive_explanation
        )
        
        # Cache explanation
        self.explanations_cache[explanation_id] = comprehensive_explanation
        
        return comprehensive_explanation
    
    def _generate_explanation_summary(self, explanation: Dict) -> str:
        """
        Generate a human-readable summary of all explanations
        """
        summary_parts = ["🔍 **Comprehensive Fraud Analysis Report**\n"]
        
        # Prediction summary
        if explanation['model_prediction']:
            fraud_prob = explanation['model_prediction']['fraud_probability']
            prediction = explanation['model_prediction']['prediction']
            
            risk_level = "HIGH" if fraud_prob > 0.7 else "MEDIUM" if fraud_prob > 0.3 else "LOW"
            summary_parts.append(
                f"**🎯 Prediction:** {'FRAUD' if prediction else 'NOT FRAUD'} "
                f"(Confidence: {fraud_prob:.1%}, Risk Level: {risk_level})\n"
            )
        
        # SHAP insights
        if explanation['shap_explanation']:
            summary_parts.append("**📊 SHAP Analysis:**")
            summary_parts.append(explanation['shap_explanation']['explanation_text'])
            summary_parts.append("")
        
        # Counterfactual insights
        if explanation['counterfactual_explanation']:
            insights = explanation['counterfactual_explanation']['actionable_insights']
            if insights:
                summary_parts.append("**🔄 Actionable Recommendations:**")
                for insight in insights[:3]:  # Top 3 recommendations
                    summary_parts.append(f"• {insight}")
                summary_parts.append("")
        
        # Compliance note
        summary_parts.append(
            "**⚖️ Compliance Note:** This explanation is generated using interpretable AI "
            "methods (SHAP, LIME, Counterfactuals) to ensure transparency and regulatory compliance."
        )
        
        return "\n".join(summary_parts)
    
    def generate_model_report(self, 
                            X_test: pd.DataFrame, 
                            y_test: pd.Series = None,
                            sample_explanations: int = 5) -> Dict:
        """
        Generate comprehensive model interpretability report
        
        Args:
            X_test: Test dataset
            y_test: Test targets (if available)
            sample_explanations: Number of sample explanations to include
            
        Returns:
            Comprehensive model report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'type': self.model.__class__.__name__,
                'features': X_test.columns.tolist()
            },
            'global_explanations': {},
            'sample_explanations': [],
            'model_performance': None
        }
        
        # Global SHAP explanations
        if self.shap_explainer:
            try:
                # Sample data for global explanations
                sample_size = min(100, len(X_test))
                sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
                X_sample = X_test.iloc[sample_indices]
                
                shap_values = self.shap_explainer.explain_predictions(X_sample)
                feature_importance = self.shap_explainer.get_feature_importance()
                
                report['global_explanations']['shap'] = {
                    'feature_importance': feature_importance.head(20).to_dict('records'),
                    'top_features': feature_importance.head(10)['feature'].tolist()
                }
            except Exception as e:
                logger.warning(f"Global SHAP analysis failed: {e}")
        
        # Sample individual explanations
        sample_indices = np.random.choice(len(X_test), sample_explanations, replace=False)
        for idx in sample_indices:
            try:
                instance = X_test.iloc[idx]
                explanation = self.get_comprehensive_explanation(
                    instance, X_test, include_counterfactuals=False, include_lime=False
                )
                
                report['sample_explanations'].append({
                    'index': int(idx),
                    'prediction': explanation['model_prediction'],
                    'summary': explanation['summary']
                })
            except Exception as e:
                logger.warning(f"Sample explanation {idx} failed: {e}")
        
        # Model performance (if targets available)
        if y_test is not None:
            try:
                y_pred = self.model.predict(X_test)
                y_pred_proba = self.model.predict_proba(X_test)[:, 1]
                
                report['model_performance'] = {
                    'classification_report': classification_report(y_test, y_pred, output_dict=True),
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                    'auc_score': float(sklearn.metrics.roc_auc_score(y_test, y_pred_proba))
                }
            except Exception as e:
                logger.warning(f"Performance evaluation failed: {e}")
        
        return report
    
    def save_explanations(self, explanations: Dict, filepath: str) -> None:
        """
        Save explanations to file
        
        Args:
            explanations: Explanations dictionary
            filepath: Path to save file
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(explanations, f, indent=2, default=str)
            logger.info(f"Explanations saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save explanations: {e}")
    
    def load_explanations(self, filepath: str) -> Dict:
        """
        Load explanations from file
        
        Args:
            filepath: Path to load file
            
        Returns:
            Loaded explanations dictionary
        """
        try:
            with open(filepath, 'r') as f:
                explanations = json.load(f)
            logger.info(f"Explanations loaded from {filepath}")
            return explanations
        except Exception as e:
            logger.error(f"Failed to load explanations: {e}")
            return {}