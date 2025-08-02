"""
Real-time Fraud Detection Engine
Orchestrates Hub and Spoke models for production fraud scoring.
"""

import logging
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import redis
import json

from ..models.hub_model import HubModelManager
from ..models.spoke_models import SpokeModelManager
from ..data.schemas import (
    Transaction, ModelPrediction, RiskLevel, 
    ProductType, ThresholdConfig
)
from ..features.feature_store import FeatureStore
from ..infrastructure.database import DatabaseManager
from ..utils.config_manager import ConfigManager


logger = logging.getLogger(__name__)


class DecisionAction(str, Enum):
    """Possible actions for fraud detection"""
    APPROVE = "approve"
    REJECT = "reject"
    CHALLENGE = "challenge"
    MANUAL_REVIEW = "manual_review"


@dataclass
class FraudDetectionRequest:
    """Request for fraud detection"""
    transaction_id: str
    customer_id: str
    product_type: ProductType
    transaction_data: Dict[str, Any]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class FraudDetectionResponse:
    """Response from fraud detection"""
    transaction_id: str
    customer_id: str
    product_type: ProductType
    
    # Scores
    hub_risk_score: float
    spoke_fraud_score: float
    final_score: float
    
    # Classification
    risk_level: RiskLevel
    predicted_class: str
    confidence: float
    
    # Decision
    action: DecisionAction
    reason_codes: List[str]
    
    # Metadata
    processing_time_ms: float
    model_versions: Dict[str, str]
    feature_importance: Dict[str, float]
    timestamp: datetime
    
    # Explanation
    explanation: Dict[str, Any]


class RiskThresholdManager:
    """Manages risk thresholds and decision logic"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.thresholds = self._load_thresholds()
    
    def _load_thresholds(self) -> Dict[str, ThresholdConfig]:
        """Load risk thresholds from configuration"""
        
        thresholds = {}
        
        # Default thresholds
        default_config = ThresholdConfig(
            model_name="default",
            low_risk_threshold=0.2,
            medium_risk_threshold=0.5,
            high_risk_threshold=0.8,
            low_risk_action=DecisionAction.APPROVE.value,
            medium_risk_action=DecisionAction.CHALLENGE.value,
            high_risk_action=DecisionAction.REJECT.value,
            updated_at=datetime.now()
        )
        
        thresholds["default"] = default_config
        
        # Product-specific thresholds
        for product_type in ProductType:
            product_config = ThresholdConfig(
                model_name=f"{product_type.value}_model",
                low_risk_threshold=self.config.get(f"inference.thresholds.{product_type.value}.low_risk", 0.2),
                medium_risk_threshold=self.config.get(f"inference.thresholds.{product_type.value}.medium_risk", 0.5),
                high_risk_threshold=self.config.get(f"inference.thresholds.{product_type.value}.high_risk", 0.8),
                low_risk_action=DecisionAction.APPROVE.value,
                medium_risk_action=DecisionAction.CHALLENGE.value,
                high_risk_action=DecisionAction.REJECT.value,
                updated_at=datetime.now()
            )
            thresholds[product_type.value] = product_config
        
        return thresholds
    
    def get_risk_level(self, score: float, product_type: str = "default") -> RiskLevel:
        """Determine risk level based on score and thresholds"""
        
        threshold_config = self.thresholds.get(product_type, self.thresholds["default"])
        
        if score >= threshold_config.high_risk_threshold:
            return RiskLevel.HIGH
        elif score >= threshold_config.medium_risk_threshold:
            return RiskLevel.MEDIUM
        elif score >= threshold_config.low_risk_threshold:
            return RiskLevel.LOW
        else:
            return RiskLevel.LOW
    
    def get_action(self, risk_level: RiskLevel, product_type: str = "default") -> DecisionAction:
        """Get decision action based on risk level"""
        
        threshold_config = self.thresholds.get(product_type, self.thresholds["default"])
        
        action_mapping = {
            RiskLevel.LOW: threshold_config.low_risk_action,
            RiskLevel.MEDIUM: threshold_config.medium_risk_action,
            RiskLevel.HIGH: threshold_config.high_risk_action,
            RiskLevel.CRITICAL: DecisionAction.REJECT.value
        }
        
        return DecisionAction(action_mapping.get(risk_level, DecisionAction.MANUAL_REVIEW.value))


class FeatureCacheManager:
    """Manages caching of features for performance optimization"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.default_ttl = 300  # 5 minutes
    
    def get_cached_hub_score(self, customer_id: str) -> Optional[float]:
        """Get cached hub risk score"""
        
        cache_key = f"hub_score:{customer_id}"
        
        try:
            cached_score = self.redis_client.get(cache_key)
            if cached_score:
                return float(cached_score)
        except Exception as e:
            logger.error(f"Error getting cached hub score: {e}")
        
        return None
    
    def cache_hub_score(self, customer_id: str, score: float, ttl: int = None):
        """Cache hub risk score"""
        
        cache_key = f"hub_score:{customer_id}"
        ttl = ttl or self.default_ttl
        
        try:
            self.redis_client.setex(cache_key, ttl, str(score))
        except Exception as e:
            logger.error(f"Error caching hub score: {e}")
    
    def invalidate_customer_cache(self, customer_id: str):
        """Invalidate all cached data for customer"""
        
        patterns = [
            f"hub_score:{customer_id}",
            f"features:*:{customer_id}"
        ]
        
        for pattern in patterns:
            try:
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
            except Exception as e:
                logger.error(f"Error invalidating cache: {e}")


class FraudDetectionEngine:
    """
    Main fraud detection engine that orchestrates Hub and Spoke models.
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.db = DatabaseManager(config_manager)
        self.feature_store = FeatureStore(config_manager)
        self.hub_model_manager = HubModelManager(config_manager)
        self.spoke_model_manager = SpokeModelManager(config_manager)
        self.threshold_manager = RiskThresholdManager(config_manager)
        
        # Redis for caching
        redis_url = config_manager.get("feature_store.online_store.connection_string")
        self.redis_client = redis.Redis.from_url(redis_url)
        self.cache_manager = FeatureCacheManager(self.redis_client)
        
        # Performance metrics
        self.request_count = 0
        self.total_processing_time = 0.0
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models"""
        
        try:
            # Load hub model
            hub_model_path = self.config.get("models.hub_model.model_path")
            if hub_model_path:
                self.hub_model_manager.load_model(hub_model_path)
                logger.info("Hub model loaded successfully")
            
            # Load spoke models
            spoke_model_paths = self.config.get("models.spoke_models", {})
            for product_type, model_path in spoke_model_paths.items():
                self.spoke_model_manager.load_spoke_model(product_type, model_path)
                logger.info(f"{product_type} spoke model loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    async def detect_fraud(self, request: FraudDetectionRequest) -> FraudDetectionResponse:
        """
        Main fraud detection method - orchestrates the full process.
        """
        
        start_time = time.time()
        
        try:
            # Step 1: Get Hub model risk score (unified customer risk)
            hub_score = await self._get_hub_risk_score(request.customer_id)
            
            # Step 2: Get Spoke model fraud score (product-specific)
            spoke_result = await self._get_spoke_fraud_score(request)
            
            # Step 3: Combine scores using weighted approach
            final_score = self._combine_scores(hub_score, spoke_result['fraud_score'])
            
            # Step 4: Determine risk level and action
            risk_level = self.threshold_manager.get_risk_level(
                final_score, request.product_type.value
            )
            action = self.threshold_manager.get_action(
                risk_level, request.product_type.value
            )
            
            # Step 5: Generate reason codes
            reason_codes = self._generate_reason_codes(
                hub_score, spoke_result, final_score, risk_level
            )
            
            # Step 6: Create response
            processing_time = (time.time() - start_time) * 1000
            
            response = FraudDetectionResponse(
                transaction_id=request.transaction_id,
                customer_id=request.customer_id,
                product_type=request.product_type,
                
                hub_risk_score=hub_score,
                spoke_fraud_score=spoke_result['fraud_score'],
                final_score=final_score,
                
                risk_level=risk_level,
                predicted_class="fraud" if final_score > 0.5 else "legitimate",
                confidence=max(final_score, 1 - final_score),
                
                action=action,
                reason_codes=reason_codes,
                
                processing_time_ms=processing_time,
                model_versions={
                    'hub_model': self.hub_model_manager.model_metadata.get('version', 'unknown'),
                    'spoke_model': spoke_result.get('model_version', 'unknown')
                },
                feature_importance=spoke_result.get('feature_importance', {}),
                timestamp=datetime.now(),
                
                explanation=self._generate_explanation(hub_score, spoke_result, final_score)
            )
            
            # Step 7: Log prediction
            await self._log_prediction(request, response)
            
            # Update metrics
            self.request_count += 1
            self.total_processing_time += processing_time
            
            return response
            
        except Exception as e:
            logger.error(f"Error in fraud detection: {e}")
            
            # Return safe default response
            return self._create_error_response(request, str(e), time.time() - start_time)
    
    async def _get_hub_risk_score(self, customer_id: str) -> float:
        """Get unified risk score from Hub model"""
        
        # Check cache first
        cached_score = self.cache_manager.get_cached_hub_score(customer_id)
        if cached_score is not None:
            return cached_score
        
        try:
            # Get fresh score from Hub model
            hub_score = self.hub_model_manager.predict_risk_score(customer_id)
            
            # Cache the score
            self.cache_manager.cache_hub_score(customer_id, hub_score)
            
            return hub_score
            
        except Exception as e:
            logger.error(f"Error getting hub risk score for {customer_id}: {e}")
            return 0.5  # Safe default
    
    async def _get_spoke_fraud_score(self, request: FraudDetectionRequest) -> Dict[str, Any]:
        """Get fraud score from appropriate Spoke model"""
        
        try:
            # Create transaction context
            transaction_context = {
                'transaction': request.transaction_data
            }
            
            # Get prediction from spoke model
            spoke_prediction = self.spoke_model_manager.predict(
                request.product_type.value,
                request.customer_id,
                transaction_context
            )
            
            return {
                'fraud_score': spoke_prediction['fraud_probability'],
                'feature_importance': spoke_prediction.get('contextual_features', {}),
                'model_version': spoke_prediction.get('model_name', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Error getting spoke fraud score: {e}")
            return {
                'fraud_score': 0.5,  # Safe default
                'feature_importance': {},
                'model_version': 'error'
            }
    
    def _combine_scores(self, hub_score: float, spoke_score: float) -> float:
        """
        Combine Hub and Spoke scores using weighted approach.
        Hub score represents overall customer risk.
        Spoke score represents transaction-specific risk.
        """
        
        # Weighted combination: Hub model carries customer context weight
        # Spoke model carries transaction context weight
        hub_weight = 0.4
        spoke_weight = 0.6
        
        # Non-linear combination to handle extreme cases
        combined_score = (hub_weight * hub_score + spoke_weight * spoke_score)
        
        # Boost if both models agree on high risk
        if hub_score > 0.7 and spoke_score > 0.7:
            combined_score = min(1.0, combined_score * 1.2)
        
        # Dampen if hub model shows very low risk but spoke shows high risk
        # (could be false positive)
        elif hub_score < 0.2 and spoke_score > 0.8:
            combined_score = combined_score * 0.8
        
        return max(0.0, min(1.0, combined_score))
    
    def _generate_reason_codes(
        self, 
        hub_score: float, 
        spoke_result: Dict[str, Any], 
        final_score: float,
        risk_level: RiskLevel
    ) -> List[str]:
        """Generate human-readable reason codes"""
        
        reason_codes = []
        spoke_score = spoke_result['fraud_score']
        
        # High-level risk reasons
        if risk_level == RiskLevel.HIGH:
            reason_codes.append("HIGH_RISK_TRANSACTION")
        
        if hub_score > 0.7:
            reason_codes.append("HIGH_CUSTOMER_RISK_PROFILE")
        
        if spoke_score > 0.7:
            reason_codes.append("SUSPICIOUS_TRANSACTION_PATTERN")
        
        # Specific feature-based reasons
        features = spoke_result.get('feature_importance', {})
        
        if features.get('is_night_transaction', 0) == 1:
            reason_codes.append("UNUSUAL_TIME_OF_DAY")
        
        if features.get('distance_from_usual_location_km', 0) > 100:
            reason_codes.append("UNUSUAL_LOCATION")
        
        if features.get('amount_zscore', 0) > 2:
            reason_codes.append("UNUSUAL_AMOUNT")
        
        if features.get('is_new_beneficiary', 0) == 1:
            reason_codes.append("NEW_BENEFICIARY")
        
        if features.get('fraudulent_beneficiaries_count', 0) > 0:
            reason_codes.append("RISKY_BENEFICIARY_NETWORK")
        
        if features.get('customers_sharing_devices', 0) > 5:
            reason_codes.append("SHARED_DEVICE_RISK")
        
        return reason_codes[:5]  # Limit to top 5 reasons
    
    def _generate_explanation(
        self, 
        hub_score: float, 
        spoke_result: Dict[str, Any], 
        final_score: float
    ) -> Dict[str, Any]:
        """Generate detailed explanation for the decision"""
        
        return {
            'summary': f"Customer risk level: {hub_score:.2f}, Transaction risk: {spoke_result['fraud_score']:.2f}",
            'hub_model_contribution': f"{hub_score * 0.4:.3f} (40% weight)",
            'spoke_model_contribution': f"{spoke_result['fraud_score'] * 0.6:.3f} (60% weight)",
            'final_score_calculation': f"Final score: {final_score:.3f}",
            'key_risk_factors': list(spoke_result.get('feature_importance', {}).keys())[:3]
        }
    
    async def _log_prediction(self, request: FraudDetectionRequest, response: FraudDetectionResponse):
        """Log prediction for monitoring and model improvement"""
        
        try:
            prediction_log = {
                'transaction_id': request.transaction_id,
                'customer_id': request.customer_id,
                'product_type': request.product_type.value,
                'hub_score': response.hub_risk_score,
                'spoke_score': response.spoke_fraud_score,
                'final_score': response.final_score,
                'risk_level': response.risk_level.value,
                'action': response.action.value,
                'processing_time_ms': response.processing_time_ms,
                'timestamp': response.timestamp.isoformat(),
                'model_versions': response.model_versions
            }
            
            # Log to database for analytics
            await self._save_prediction_log(prediction_log)
            
            # Send to monitoring system
            await self._send_monitoring_metrics(response)
            
        except Exception as e:
            logger.error(f"Error logging prediction: {e}")
    
    async def _save_prediction_log(self, prediction_log: Dict[str, Any]):
        """Save prediction log to database"""
        
        try:
            # Insert into predictions table
            query = """
                INSERT INTO fraud_predictions 
                (transaction_id, customer_id, product_type, hub_score, spoke_score, 
                 final_score, risk_level, action, processing_time_ms, timestamp, model_versions)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            params = [
                prediction_log['transaction_id'],
                prediction_log['customer_id'],
                prediction_log['product_type'],
                prediction_log['hub_score'],
                prediction_log['spoke_score'],
                prediction_log['final_score'],
                prediction_log['risk_level'],
                prediction_log['action'],
                prediction_log['processing_time_ms'],
                prediction_log['timestamp'],
                json.dumps(prediction_log['model_versions'])
            ]
            
            self.db.execute_query(query, params=params)
            
        except Exception as e:
            logger.error(f"Error saving prediction log: {e}")
    
    async def _send_monitoring_metrics(self, response: FraudDetectionResponse):
        """Send metrics to monitoring system"""
        
        try:
            # Send metrics to Redis for monitoring dashboard
            metrics = {
                'timestamp': response.timestamp.isoformat(),
                'processing_time_ms': response.processing_time_ms,
                'risk_level': response.risk_level.value,
                'action': response.action.value,
                'final_score': response.final_score
            }
            
            # Publish to monitoring channel
            self.redis_client.publish('fraud_detection_metrics', json.dumps(metrics))
            
        except Exception as e:
            logger.error(f"Error sending monitoring metrics: {e}")
    
    def _create_error_response(
        self, 
        request: FraudDetectionRequest, 
        error_message: str, 
        processing_time: float
    ) -> FraudDetectionResponse:
        """Create error response with safe defaults"""
        
        return FraudDetectionResponse(
            transaction_id=request.transaction_id,
            customer_id=request.customer_id,
            product_type=request.product_type,
            
            hub_risk_score=0.5,
            spoke_fraud_score=0.5,
            final_score=0.5,
            
            risk_level=RiskLevel.MEDIUM,
            predicted_class="unknown",
            confidence=0.0,
            
            action=DecisionAction.MANUAL_REVIEW,
            reason_codes=["SYSTEM_ERROR", "MANUAL_REVIEW_REQUIRED"],
            
            processing_time_ms=processing_time * 1000,
            model_versions={'error': error_message},
            feature_importance={},
            timestamp=datetime.now(),
            
            explanation={'error': error_message}
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get engine performance metrics"""
        
        avg_processing_time = (
            self.total_processing_time / self.request_count 
            if self.request_count > 0 else 0
        )
        
        return {
            'total_requests': self.request_count,
            'average_processing_time_ms': avg_processing_time,
            'total_processing_time_ms': self.total_processing_time,
            'requests_per_second': self.request_count / max(1, self.total_processing_time / 1000),
            'timestamp': datetime.now().isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the fraud detection engine"""
        
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        # Check database connection
        try:
            self.db.execute_query("SELECT 1")
            health_status['components']['database'] = 'healthy'
        except Exception as e:
            health_status['components']['database'] = f'unhealthy: {e}'
            health_status['status'] = 'unhealthy'
        
        # Check Redis connection
        try:
            self.redis_client.ping()
            health_status['components']['redis'] = 'healthy'
        except Exception as e:
            health_status['components']['redis'] = f'unhealthy: {e}'
            health_status['status'] = 'unhealthy'
        
        # Check models
        health_status['components']['hub_model'] = (
            'loaded' if self.hub_model_manager.model and 
            self.hub_model_manager.model.is_trained else 'not_loaded'
        )
        
        health_status['components']['spoke_models'] = {
            product_type: 'loaded' if model.is_trained else 'not_loaded'
            for product_type, model in self.spoke_model_manager.spoke_models.items()
        }
        
        return health_status