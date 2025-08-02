"""
Feature Store Implementation - 4-Pillar Feature Architecture
Implements enterprise-grade feature engineering and serving for fraud detection.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import redis
import hashlib
import json
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..data.schemas import FeatureVector, CustomerProfile, Transaction
from ..infrastructure.database import DatabaseManager
from ..utils.config_manager import ConfigManager


logger = logging.getLogger(__name__)


@dataclass
class FeatureDefinition:
    """Definition of a feature"""
    name: str
    feature_type: str  # profile, behavioral, network, contextual
    data_type: str  # int, float, bool, string
    description: str
    computation_query: str
    dependencies: List[str] = None
    ttl_seconds: int = 3600  # Time to live in cache
    refresh_frequency: str = "daily"  # hourly, daily, weekly


class FeatureComputer(ABC):
    """Abstract base class for feature computation"""
    
    @abstractmethod
    def compute_features(self, customer_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Compute features for a customer"""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get list of feature names this computer produces"""
        pass


class ProfileFeatureComputer(FeatureComputer):
    """
    Pillar 1: Profile Features (Static/Slow-changing)
    Features that describe who the customer is.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.feature_definitions = self._load_feature_definitions()
    
    def _load_feature_definitions(self) -> List[FeatureDefinition]:
        """Load profile feature definitions"""
        return [
            FeatureDefinition(
                name="customer_age",
                feature_type="profile",
                data_type="int",
                description="Customer age in years",
                computation_query="SELECT EXTRACT(YEAR FROM AGE(birth_date)) as customer_age FROM customers WHERE customer_id = %s"
            ),
            FeatureDefinition(
                name="account_age_days",
                feature_type="profile", 
                data_type="int",
                description="Days since account opening",
                computation_query="SELECT EXTRACT(EPOCH FROM (NOW() - created_at))::int / 86400 as account_age_days FROM customers WHERE customer_id = %s"
            ),
            FeatureDefinition(
                name="total_products_count",
                feature_type="profile",
                data_type="int", 
                description="Total number of products customer has",
                computation_query="SELECT COUNT(*) as total_products_count FROM customer_products WHERE customer_id = %s AND status = 'active'"
            ),
            FeatureDefinition(
                name="credit_score_internal",
                feature_type="profile",
                data_type="int",
                description="Internal credit score",
                computation_query="SELECT credit_score_internal FROM customer_risk_profile WHERE customer_id = %s"
            ),
            FeatureDefinition(
                name="is_pep",
                feature_type="profile",
                data_type="bool",
                description="Is politically exposed person",
                computation_query="SELECT is_pep FROM customer_risk_profile WHERE customer_id = %s"
            ),
            FeatureDefinition(
                name="income_bracket_encoded",
                feature_type="profile",
                data_type="int",
                description="Income bracket encoded (1=low, 2=medium, 3=high)",
                computation_query="""
                    SELECT CASE 
                        WHEN income_bracket = 'low' THEN 1
                        WHEN income_bracket = 'medium' THEN 2 
                        WHEN income_bracket = 'high' THEN 3
                        ELSE 0 END as income_bracket_encoded
                    FROM customers WHERE customer_id = %s
                """
            ),
            FeatureDefinition(
                name="kyc_completion_score",
                feature_type="profile",
                data_type="float",
                description="KYC completion percentage",
                computation_query="""
                    SELECT 
                        (CASE WHEN full_name IS NOT NULL THEN 0.2 ELSE 0 END +
                         CASE WHEN email IS NOT NULL THEN 0.2 ELSE 0 END +
                         CASE WHEN phone IS NOT NULL THEN 0.2 ELSE 0 END +
                         CASE WHEN address IS NOT NULL THEN 0.2 ELSE 0 END +
                         CASE WHEN income_verified THEN 0.2 ELSE 0 END) as kyc_completion_score
                    FROM customers WHERE customer_id = %s
                """
            )
        ]
    
    def compute_features(self, customer_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Compute profile features for customer"""
        
        features = {}
        
        for feature_def in self.feature_definitions:
            try:
                result = self.db.execute_query(feature_def.computation_query, params=[customer_id])
                
                if len(result) > 0:
                    value = result.iloc[0][feature_def.name]
                    
                    # Handle null values
                    if pd.isna(value):
                        if feature_def.data_type == "int":
                            value = 0
                        elif feature_def.data_type == "float":
                            value = 0.0
                        elif feature_def.data_type == "bool":
                            value = False
                        else:
                            value = ""
                    
                    features[feature_def.name] = value
                else:
                    # Default values for missing data
                    features[feature_def.name] = self._get_default_value(feature_def.data_type)
                    
            except Exception as e:
                logger.error(f"Error computing feature {feature_def.name}: {e}")
                features[feature_def.name] = self._get_default_value(feature_def.data_type)
        
        return features
    
    def _get_default_value(self, data_type: str):
        """Get default value for data type"""
        defaults = {
            "int": 0,
            "float": 0.0,
            "bool": False,
            "string": ""
        }
        return defaults.get(data_type, None)
    
    def get_feature_names(self) -> List[str]:
        """Get list of profile feature names"""
        return [fd.name for fd in self.feature_definitions]


class BehavioralFeatureComputer(FeatureComputer):
    """
    Pillar 2: Behavioral Features (Cross-Product Aggregations)
    Features that describe how the customer behaves across all products.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.time_windows = ["1h", "6h", "24h", "7d", "30d", "90d"]
        self.aggregations = ["sum", "mean", "count", "std", "min", "max"]
    
    def compute_features(self, customer_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Compute behavioral features for customer"""
        
        features = {}
        
        # Transaction volume features
        transaction_features = self._compute_transaction_features(customer_id)
        features.update(transaction_features)
        
        # Channel usage features
        channel_features = self._compute_channel_features(customer_id)
        features.update(channel_features)
        
        # Digital behavior features
        digital_features = self._compute_digital_behavior_features(customer_id)
        features.update(digital_features)
        
        # Velocity features
        velocity_features = self._compute_velocity_features(customer_id)
        features.update(velocity_features)
        
        return features
    
    def _compute_transaction_features(self, customer_id: str) -> Dict[str, Any]:
        """Compute transaction-based behavioral features"""
        
        features = {}
        
        for window in self.time_windows:
            window_hours = self._convert_window_to_hours(window)
            
            # Transaction volume and count by time window
            query = f"""
                SELECT 
                    COUNT(*) as transaction_count_{window},
                    COALESCE(SUM(amount), 0) as transaction_volume_{window},
                    COALESCE(AVG(amount), 0) as avg_transaction_amount_{window},
                    COALESCE(STDDEV(amount), 0) as std_transaction_amount_{window},
                    COUNT(DISTINCT product_type) as products_used_{window},
                    COUNT(DISTINCT DATE(timestamp)) as active_days_{window}
                FROM transactions 
                WHERE customer_id = %s 
                AND timestamp >= NOW() - INTERVAL '{window_hours} hours'
                AND status = 'approved'
            """
            
            try:
                result = self.db.execute_query(query, params=[customer_id])
                if len(result) > 0:
                    row = result.iloc[0]
                    for col in result.columns:
                        features[col] = float(row[col]) if pd.notna(row[col]) else 0.0
            except Exception as e:
                logger.error(f"Error computing transaction features for {window}: {e}")
        
        return features
    
    def _compute_channel_features(self, customer_id: str) -> Dict[str, Any]:
        """Compute channel usage behavioral features"""
        
        features = {}
        
        for window in self.time_windows:
            window_hours = self._convert_window_to_hours(window)
            
            # Channel diversity and usage patterns
            query = f"""
                SELECT 
                    COUNT(DISTINCT channel) as channels_used_{window},
                    COUNT(DISTINCT device_id) as devices_used_{window},
                    COUNT(DISTINCT ip_address) as ips_used_{window},
                    SUM(CASE WHEN channel = 'mobile_app' THEN 1 ELSE 0 END) as mobile_transactions_{window},
                    SUM(CASE WHEN channel = 'web_browser' THEN 1 ELSE 0 END) as web_transactions_{window},
                    SUM(CASE WHEN channel = 'atm' THEN 1 ELSE 0 END) as atm_transactions_{window}
                FROM transactions 
                WHERE customer_id = %s 
                AND timestamp >= NOW() - INTERVAL '{window_hours} hours'
            """
            
            try:
                result = self.db.execute_query(query, params=[customer_id])
                if len(result) > 0:
                    row = result.iloc[0]
                    for col in result.columns:
                        features[col] = int(row[col]) if pd.notna(row[col]) else 0
            except Exception as e:
                logger.error(f"Error computing channel features for {window}: {e}")
        
        return features
    
    def _compute_digital_behavior_features(self, customer_id: str) -> Dict[str, Any]:
        """Compute digital behavior features from app/web usage"""
        
        features = {}
        
        for window in self.time_windows:
            window_hours = self._convert_window_to_hours(window)
            
            # Digital engagement patterns
            query = f"""
                SELECT 
                    COUNT(*) as login_count_{window},
                    COUNT(DISTINCT DATE(timestamp)) as login_days_{window},
                    AVG(session_duration_minutes) as avg_session_duration_{window},
                    SUM(CASE WHEN event_type = 'password_change' THEN 1 ELSE 0 END) as password_changes_{window},
                    SUM(CASE WHEN event_type = 'profile_update' THEN 1 ELSE 0 END) as profile_updates_{window},
                    COUNT(DISTINCT EXTRACT(HOUR FROM timestamp)) as active_hours_{window}
                FROM customer_events 
                WHERE customer_id = %s 
                AND timestamp >= NOW() - INTERVAL '{window_hours} hours'
            """
            
            try:
                result = self.db.execute_query(query, params=[customer_id])
                if len(result) > 0:
                    row = result.iloc[0]
                    for col in result.columns:
                        features[col] = float(row[col]) if pd.notna(row[col]) else 0.0
            except Exception as e:
                logger.error(f"Error computing digital behavior features for {window}: {e}")
        
        return features
    
    def _compute_velocity_features(self, customer_id: str) -> Dict[str, Any]:
        """Compute velocity and frequency features"""
        
        features = {}
        
        # Transaction velocity (transactions per hour in different windows)
        for window in ["1h", "6h", "24h"]:
            window_hours = self._convert_window_to_hours(window)
            
            query = f"""
                SELECT 
                    COUNT(*)::float / {window_hours} as transaction_velocity_{window},
                    COALESCE(SUM(amount)::float / {window_hours}, 0) as volume_velocity_{window}
                FROM transactions 
                WHERE customer_id = %s 
                AND timestamp >= NOW() - INTERVAL '{window_hours} hours'
                AND status = 'approved'
            """
            
            try:
                result = self.db.execute_query(query, params=[customer_id])
                if len(result) > 0:
                    row = result.iloc[0]
                    for col in result.columns:
                        features[col] = float(row[col]) if pd.notna(row[col]) else 0.0
            except Exception as e:
                logger.error(f"Error computing velocity features for {window}: {e}")
        
        return features
    
    def _convert_window_to_hours(self, window: str) -> int:
        """Convert time window string to hours"""
        conversions = {
            "1h": 1,
            "6h": 6, 
            "24h": 24,
            "7d": 168,
            "30d": 720,
            "90d": 2160
        }
        return conversions.get(window, 24)
    
    def get_feature_names(self) -> List[str]:
        """Get list of behavioral feature names"""
        # This would return a comprehensive list of all behavioral features
        # For brevity, returning a subset
        return [
            f"transaction_count_{w}" for w in self.time_windows
        ] + [
            f"transaction_volume_{w}" for w in self.time_windows
        ] + [
            f"channels_used_{w}" for w in self.time_windows
        ]


class NetworkFeatureComputer(FeatureComputer):
    """
    Pillar 3: Network/Graph Features
    Features that describe relationships and network position.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def compute_features(self, customer_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Compute network/graph features for customer"""
        
        features = {}
        
        # Device sharing features
        device_features = self._compute_device_sharing_features(customer_id)
        features.update(device_features)
        
        # Beneficiary network features
        beneficiary_features = self._compute_beneficiary_network_features(customer_id)
        features.update(beneficiary_features)
        
        # Graph centrality features
        centrality_features = self._compute_centrality_features(customer_id)
        features.update(centrality_features)
        
        return features
    
    def _compute_device_sharing_features(self, customer_id: str) -> Dict[str, Any]:
        """Compute device sharing network features"""
        
        features = {}
        
        # Count customers sharing devices
        query = """
            WITH customer_devices AS (
                SELECT DISTINCT device_id 
                FROM transactions 
                WHERE customer_id = %s 
                AND device_id IS NOT NULL
                AND timestamp >= NOW() - INTERVAL '90 days'
            )
            SELECT 
                COUNT(DISTINCT t.customer_id) - 1 as customers_sharing_devices,
                COUNT(DISTINCT t.device_id) as unique_devices_used,
                MAX(device_customer_count.customer_count) as max_customers_per_device
            FROM transactions t
            JOIN customer_devices cd ON t.device_id = cd.device_id
            LEFT JOIN (
                SELECT device_id, COUNT(DISTINCT customer_id) as customer_count
                FROM transactions 
                WHERE timestamp >= NOW() - INTERVAL '90 days'
                GROUP BY device_id
            ) device_customer_count ON t.device_id = device_customer_count.device_id
            WHERE t.timestamp >= NOW() - INTERVAL '90 days'
        """
        
        try:
            result = self.db.execute_query(query, params=[customer_id])
            if len(result) > 0:
                row = result.iloc[0]
                features.update({
                    "customers_sharing_devices": int(row["customers_sharing_devices"] or 0),
                    "unique_devices_used": int(row["unique_devices_used"] or 0),
                    "max_customers_per_device": int(row["max_customers_per_device"] or 1)
                })
        except Exception as e:
            logger.error(f"Error computing device sharing features: {e}")
            features.update({
                "customers_sharing_devices": 0,
                "unique_devices_used": 0,
                "max_customers_per_device": 1
            })
        
        return features
    
    def _compute_beneficiary_network_features(self, customer_id: str) -> Dict[str, Any]:
        """Compute beneficiary network features"""
        
        features = {}
        
        # Beneficiary relationship analysis
        query = """
            WITH customer_beneficiaries AS (
                SELECT DISTINCT beneficiary_id, beneficiary_document
                FROM transactions 
                WHERE customer_id = %s 
                AND beneficiary_id IS NOT NULL
                AND timestamp >= NOW() - INTERVAL '180 days'
            ),
            beneficiary_risk AS (
                SELECT 
                    cb.beneficiary_id,
                    COUNT(DISTINCT t.customer_id) as senders_count,
                    CASE WHEN fr.customer_id IS NOT NULL THEN 1 ELSE 0 END as is_known_fraudster
                FROM customer_beneficiaries cb
                LEFT JOIN transactions t ON cb.beneficiary_id = t.beneficiary_id
                LEFT JOIN fraud_records fr ON cb.beneficiary_id = fr.customer_id
                WHERE t.timestamp >= NOW() - INTERVAL '180 days'
                GROUP BY cb.beneficiary_id, fr.customer_id
            )
            SELECT 
                COUNT(*) as unique_beneficiaries,
                AVG(br.senders_count) as avg_senders_per_beneficiary,
                MAX(br.senders_count) as max_senders_per_beneficiary,
                SUM(br.is_known_fraudster) as fraudulent_beneficiaries_count
            FROM customer_beneficiaries cb
            LEFT JOIN beneficiary_risk br ON cb.beneficiary_id = br.beneficiary_id
        """
        
        try:
            result = self.db.execute_query(query, params=[customer_id])
            if len(result) > 0:
                row = result.iloc[0]
                features.update({
                    "unique_beneficiaries": int(row["unique_beneficiaries"] or 0),
                    "avg_senders_per_beneficiary": float(row["avg_senders_per_beneficiary"] or 0),
                    "max_senders_per_beneficiary": int(row["max_senders_per_beneficiary"] or 0),
                    "fraudulent_beneficiaries_count": int(row["fraudulent_beneficiaries_count"] or 0)
                })
        except Exception as e:
            logger.error(f"Error computing beneficiary network features: {e}")
            features.update({
                "unique_beneficiaries": 0,
                "avg_senders_per_beneficiary": 0.0,
                "max_senders_per_beneficiary": 0,
                "fraudulent_beneficiaries_count": 0
            })
        
        return features
    
    def _compute_centrality_features(self, customer_id: str) -> Dict[str, Any]:
        """Compute graph centrality features"""
        
        features = {}
        
        # Transaction network centrality
        query = """
            WITH transfer_network AS (
                SELECT customer_id as source, beneficiary_id as target, COUNT(*) as weight
                FROM transactions 
                WHERE transaction_type IN ('transfer', 'pix')
                AND timestamp >= NOW() - INTERVAL '90 days'
                AND beneficiary_id IS NOT NULL
                GROUP BY customer_id, beneficiary_id
            ),
            customer_connections AS (
                SELECT 
                    %s as customer_id,
                    COUNT(*) as outgoing_connections,
                    SUM(weight) as total_outgoing_weight
                FROM transfer_network 
                WHERE source = %s
                UNION ALL
                SELECT 
                    %s as customer_id,
                    COUNT(*) as incoming_connections,
                    SUM(weight) as total_incoming_weight
                FROM transfer_network 
                WHERE target = %s
            )
            SELECT 
                COALESCE(SUM(CASE WHEN customer_id = %s THEN outgoing_connections END), 0) as out_degree,
                COALESCE(SUM(CASE WHEN customer_id = %s THEN incoming_connections END), 0) as in_degree,
                COALESCE(SUM(CASE WHEN customer_id = %s THEN total_outgoing_weight END), 0) as out_strength,
                COALESCE(SUM(CASE WHEN customer_id = %s THEN total_incoming_weight END), 0) as in_strength
            FROM customer_connections
        """
        
        try:
            params = [customer_id] * 8
            result = self.db.execute_query(query, params=params)
            if len(result) > 0:
                row = result.iloc[0]
                features.update({
                    "network_out_degree": int(row["out_degree"] or 0),
                    "network_in_degree": int(row["in_degree"] or 0),
                    "network_out_strength": float(row["out_strength"] or 0),
                    "network_in_strength": float(row["in_strength"] or 0)
                })
        except Exception as e:
            logger.error(f"Error computing centrality features: {e}")
            features.update({
                "network_out_degree": 0,
                "network_in_degree": 0,
                "network_out_strength": 0.0,
                "network_in_strength": 0.0
            })
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of network feature names"""
        return [
            "customers_sharing_devices",
            "unique_devices_used", 
            "max_customers_per_device",
            "unique_beneficiaries",
            "avg_senders_per_beneficiary",
            "max_senders_per_beneficiary",
            "fraudulent_beneficiaries_count",
            "network_out_degree",
            "network_in_degree",
            "network_out_strength",
            "network_in_strength"
        ]


class ContextualFeatureComputer(FeatureComputer):
    """
    Pillar 4: Contextual Features (Transaction-Specific)
    Features that describe the current transaction context.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def compute_features(self, customer_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Compute contextual features for current transaction"""
        
        if not context:
            return {}
        
        features = {}
        
        # Basic transaction context
        transaction = context.get('transaction')
        if transaction:
            basic_features = self._compute_basic_contextual_features(customer_id, transaction)
            features.update(basic_features)
            
            # Time-based context
            time_features = self._compute_time_contextual_features(customer_id, transaction)
            features.update(time_features)
            
            # Location context
            location_features = self._compute_location_contextual_features(customer_id, transaction)
            features.update(location_features)
            
            # Behavioral deviation context
            deviation_features = self._compute_deviation_features(customer_id, transaction)
            features.update(deviation_features)
        
        return features
    
    def _compute_basic_contextual_features(self, customer_id: str, transaction: Dict) -> Dict[str, Any]:
        """Compute basic contextual features"""
        
        features = {
            "transaction_amount": float(transaction.get('amount', 0)),
            "is_round_amount": int(float(transaction.get('amount', 0)) % 1 == 0),
            "is_weekend": int(datetime.fromisoformat(transaction.get('timestamp')).weekday() >= 5),
            "hour_of_day": datetime.fromisoformat(transaction.get('timestamp')).hour,
            "is_night_transaction": int(
                datetime.fromisoformat(transaction.get('timestamp')).hour < 6 or 
                datetime.fromisoformat(transaction.get('timestamp')).hour > 22
            )
        }
        
        # Channel and device context
        features.update({
            "is_mobile_channel": int(transaction.get('channel') == 'mobile_app'),
            "is_web_channel": int(transaction.get('channel') == 'web_browser'),
            "is_atm_channel": int(transaction.get('channel') == 'atm')
        })
        
        return features
    
    def _compute_time_contextual_features(self, customer_id: str, transaction: Dict) -> Dict[str, Any]:
        """Compute time-based contextual features"""
        
        features = {}
        transaction_time = datetime.fromisoformat(transaction.get('timestamp'))
        
        # Time since last transaction
        query = """
            SELECT 
                EXTRACT(EPOCH FROM (%s::timestamp - MAX(timestamp)))::int / 60 as minutes_since_last_transaction,
                COUNT(*) as transactions_today
            FROM transactions 
            WHERE customer_id = %s 
            AND timestamp < %s::timestamp
            AND DATE(timestamp) = DATE(%s::timestamp)
        """
        
        try:
            result = self.db.execute_query(query, params=[
                transaction.get('timestamp'), customer_id, 
                transaction.get('timestamp'), transaction.get('timestamp')
            ])
            
            if len(result) > 0:
                row = result.iloc[0]
                features.update({
                    "minutes_since_last_transaction": int(row["minutes_since_last_transaction"] or 1440),
                    "transactions_today": int(row["transactions_today"] or 0)
                })
        except Exception as e:
            logger.error(f"Error computing time contextual features: {e}")
            features.update({
                "minutes_since_last_transaction": 1440,
                "transactions_today": 0
            })
        
        return features
    
    def _compute_location_contextual_features(self, customer_id: str, transaction: Dict) -> Dict[str, Any]:
        """Compute location-based contextual features"""
        
        features = {}
        
        # Distance from usual locations
        if transaction.get('location_lat') and transaction.get('location_lon'):
            query = """
                WITH customer_locations AS (
                    SELECT 
                        location_lat, location_lon,
                        COUNT(*) as frequency
                    FROM transactions 
                    WHERE customer_id = %s 
                    AND location_lat IS NOT NULL 
                    AND location_lon IS NOT NULL
                    AND timestamp >= NOW() - INTERVAL '90 days'
                    GROUP BY location_lat, location_lon
                ),
                closest_location AS (
                    SELECT MIN(
                        6371 * acos(
                            cos(radians(%s)) * cos(radians(location_lat)) * 
                            cos(radians(location_lon) - radians(%s)) + 
                            sin(radians(%s)) * sin(radians(location_lat))
                        )
                    ) as min_distance_km
                    FROM customer_locations
                )
                SELECT COALESCE(min_distance_km, 1000) as distance_from_usual_location
                FROM closest_location
            """
            
            try:
                result = self.db.execute_query(query, params=[
                    customer_id,
                    transaction.get('location_lat'),
                    transaction.get('location_lon'),
                    transaction.get('location_lat')
                ])
                
                if len(result) > 0:
                    features["distance_from_usual_location_km"] = float(result.iloc[0]["distance_from_usual_location"])
                else:
                    features["distance_from_usual_location_km"] = 1000.0
                    
            except Exception as e:
                logger.error(f"Error computing location features: {e}")
                features["distance_from_usual_location_km"] = 1000.0
        else:
            features["distance_from_usual_location_km"] = 0.0
        
        return features
    
    def _compute_deviation_features(self, customer_id: str, transaction: Dict) -> Dict[str, Any]:
        """Compute behavioral deviation features"""
        
        features = {}
        amount = float(transaction.get('amount', 0))
        
        # Amount deviation from historical patterns
        query = """
            SELECT 
                AVG(amount) as avg_amount,
                STDDEV(amount) as std_amount,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY amount) as p95_amount,
                MAX(amount) as max_amount
            FROM transactions 
            WHERE customer_id = %s 
            AND timestamp >= NOW() - INTERVAL '90 days'
            AND status = 'approved'
        """
        
        try:
            result = self.db.execute_query(query, params=[customer_id])
            
            if len(result) > 0 and len(result.iloc[0]) > 0:
                row = result.iloc[0]
                avg_amount = float(row["avg_amount"] or 0)
                std_amount = float(row["std_amount"] or 1)
                p95_amount = float(row["p95_amount"] or 0)
                max_amount = float(row["max_amount"] or 0)
                
                features.update({
                    "amount_zscore": (amount - avg_amount) / max(std_amount, 1),
                    "amount_vs_p95_ratio": amount / max(p95_amount, 1),
                    "amount_vs_max_ratio": amount / max(max_amount, 1),
                    "is_amount_above_p95": int(amount > p95_amount)
                })
            else:
                features.update({
                    "amount_zscore": 0.0,
                    "amount_vs_p95_ratio": 1.0,
                    "amount_vs_max_ratio": 1.0,
                    "is_amount_above_p95": 0
                })
                
        except Exception as e:
            logger.error(f"Error computing deviation features: {e}")
            features.update({
                "amount_zscore": 0.0,
                "amount_vs_p95_ratio": 1.0,
                "amount_vs_max_ratio": 1.0,
                "is_amount_above_p95": 0
            })
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of contextual feature names"""
        return [
            "transaction_amount",
            "is_round_amount",
            "is_weekend", 
            "hour_of_day",
            "is_night_transaction",
            "is_mobile_channel",
            "is_web_channel",
            "is_atm_channel",
            "minutes_since_last_transaction",
            "transactions_today",
            "distance_from_usual_location_km",
            "amount_zscore",
            "amount_vs_p95_ratio",
            "amount_vs_max_ratio",
            "is_amount_above_p95"
        ]


class FeatureStore:
    """
    Main Feature Store class that orchestrates feature computation and serving.
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.db = DatabaseManager(config_manager)
        self.redis_client = redis.Redis.from_url(
            config_manager.get("feature_store.online_store.connection_string")
        )
        
        # Initialize feature computers
        self.profile_computer = ProfileFeatureComputer(self.db)
        self.behavioral_computer = BehavioralFeatureComputer(self.db)
        self.network_computer = NetworkFeatureComputer(self.db)
        self.contextual_computer = ContextualFeatureComputer(self.db)
        
        self.feature_computers = {
            "profile": self.profile_computer,
            "behavioral": self.behavioral_computer,
            "network": self.network_computer,
            "contextual": self.contextual_computer
        }
    
    def get_feature_vector(self, customer_id: str, context: Dict[str, Any] = None) -> FeatureVector:
        """Get complete feature vector for customer"""
        
        feature_vector = FeatureVector(
            customer_id=customer_id,
            timestamp=datetime.now()
        )
        
        # Compute features from each pillar
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit parallel computation tasks
            future_to_pillar = {}
            
            for pillar_name, computer in self.feature_computers.items():
                if pillar_name == "contextual":
                    # Contextual features need transaction context
                    future = executor.submit(computer.compute_features, customer_id, context)
                else:
                    # Check cache first for non-contextual features
                    cached_features = self._get_cached_features(customer_id, pillar_name)
                    if cached_features:
                        setattr(feature_vector, f"{pillar_name}_features", cached_features)
                        continue
                    else:
                        future = executor.submit(computer.compute_features, customer_id)
                
                future_to_pillar[future] = pillar_name
            
            # Collect results
            for future in as_completed(future_to_pillar):
                pillar_name = future_to_pillar[future]
                try:
                    features = future.result()
                    setattr(feature_vector, f"{pillar_name}_features", features)
                    
                    # Cache non-contextual features
                    if pillar_name != "contextual":
                        self._cache_features(customer_id, pillar_name, features)
                        
                except Exception as e:
                    logger.error(f"Error computing {pillar_name} features: {e}")
                    setattr(feature_vector, f"{pillar_name}_features", {})
        
        return feature_vector
    
    def _get_cached_features(self, customer_id: str, pillar_name: str) -> Optional[Dict[str, Any]]:
        """Get cached features for customer and pillar"""
        
        cache_key = f"features:{pillar_name}:{customer_id}"
        
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.error(f"Error getting cached features: {e}")
        
        return None
    
    def _cache_features(self, customer_id: str, pillar_name: str, features: Dict[str, Any], ttl: int = 3600):
        """Cache features for customer and pillar"""
        
        cache_key = f"features:{pillar_name}:{customer_id}"
        
        try:
            self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(features, default=str)
            )
        except Exception as e:
            logger.error(f"Error caching features: {e}")
    
    def invalidate_cache(self, customer_id: str, pillar_name: str = None):
        """Invalidate cached features"""
        
        if pillar_name:
            cache_key = f"features:{pillar_name}:{customer_id}"
            self.redis_client.delete(cache_key)
        else:
            # Invalidate all pillars for customer
            for pillar in self.feature_computers.keys():
                cache_key = f"features:{pillar}:{customer_id}"
                self.redis_client.delete(cache_key)
    
    def get_all_feature_names(self) -> Dict[str, List[str]]:
        """Get all feature names organized by pillar"""
        
        return {
            pillar_name: computer.get_feature_names()
            for pillar_name, computer in self.feature_computers.items()
        }
    
    def batch_compute_features(self, customer_ids: List[str], pillar_name: str = None) -> Dict[str, Dict[str, Any]]:
        """Batch compute features for multiple customers"""
        
        results = {}
        
        computers_to_run = (
            {pillar_name: self.feature_computers[pillar_name]} 
            if pillar_name 
            else self.feature_computers
        )
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            
            for customer_id in customer_ids:
                for pillar, computer in computers_to_run.items():
                    if pillar != "contextual":  # Skip contextual for batch processing
                        future = executor.submit(computer.compute_features, customer_id)
                        futures.append((future, customer_id, pillar))
            
            for future, customer_id, pillar in futures:
                try:
                    features = future.result()
                    
                    if customer_id not in results:
                        results[customer_id] = {}
                    
                    results[customer_id][f"{pillar}_features"] = features
                    
                    # Cache the results
                    self._cache_features(customer_id, pillar, features)
                    
                except Exception as e:
                    logger.error(f"Error in batch computation for {customer_id}, {pillar}: {e}")
        
        return results