"""
Configuration Management for Enterprise Fraud Detection System
Handles environment-specific configuration with validation and type safety.
"""

import os
import yaml
import json
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass
from copy import deepcopy
import re


logger = logging.getLogger(__name__)


@dataclass
class EnvironmentConfig:
    """Environment-specific configuration"""
    name: str
    debug: bool = False
    log_level: str = "INFO"
    
    # Database settings
    db_pool_size: int = 10
    db_max_overflow: int = 20
    
    # Cache settings
    cache_ttl_seconds: int = 3600
    
    # Model settings
    model_cache_enabled: bool = True
    feature_cache_enabled: bool = True
    
    # API settings
    api_rate_limit: int = 1000
    api_timeout_seconds: int = 30


class ConfigManager:
    """
    Centralized configuration manager with support for:
    - Environment-specific settings
    - Environment variable substitution
    - Configuration validation
    - Runtime configuration updates
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self.config_data = {}
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.env_configs = self._setup_environment_configs()
        
        # Load configuration
        self._load_configuration()
        
        # Validate configuration
        self._validate_configuration()
        
        logger.info(f"Configuration loaded for environment: {self.environment}")
    
    def _find_config_file(self) -> str:
        """Find configuration file in standard locations"""
        
        possible_paths = [
            "config/config.yaml",
            "config.yaml",
            os.path.expanduser("~/.fraud_detection/config.yaml"),
            "/etc/fraud_detection/config.yaml"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Return default path if none found
        return "config/config.yaml"
    
    def _setup_environment_configs(self) -> Dict[str, EnvironmentConfig]:
        """Setup environment-specific configurations"""
        
        return {
            "development": EnvironmentConfig(
                name="development",
                debug=True,
                log_level="DEBUG",
                db_pool_size=5,
                cache_ttl_seconds=300,
                api_rate_limit=100
            ),
            
            "staging": EnvironmentConfig(
                name="staging",
                debug=False,
                log_level="INFO",
                db_pool_size=10,
                cache_ttl_seconds=1800,
                api_rate_limit=500
            ),
            
            "production": EnvironmentConfig(
                name="production",
                debug=False,
                log_level="WARNING",
                db_pool_size=20,
                db_max_overflow=50,
                cache_ttl_seconds=3600,
                model_cache_enabled=True,
                feature_cache_enabled=True,
                api_rate_limit=1000,
                api_timeout_seconds=60
            )
        }
    
    def _load_configuration(self):
        """Load configuration from file"""
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as file:
                    raw_config = yaml.safe_load(file)
                
                # Substitute environment variables
                self.config_data = self._substitute_env_vars(raw_config)
                
                # Merge with environment-specific settings
                self._merge_environment_config()
                
                logger.info(f"Configuration loaded from {self.config_path}")
            else:
                logger.warning(f"Configuration file not found: {self.config_path}")
                self.config_data = self._get_default_config()
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self.config_data = self._get_default_config()
    
    def _substitute_env_vars(self, config: Any) -> Any:
        """Recursively substitute environment variables in configuration"""
        
        if isinstance(config, dict):
            return {key: self._substitute_env_vars(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            return self._substitute_string_env_vars(config)
        else:
            return config
    
    def _substitute_string_env_vars(self, value: str) -> str:
        """Substitute environment variables in string values"""
        
        # Pattern: ${VAR_NAME:default_value} or ${VAR_NAME}
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        
        def replace_match(match):
            var_name = match.group(1)
            default_value = match.group(2) or ""
            
            return os.getenv(var_name, default_value)
        
        return re.sub(pattern, replace_match, value)
    
    def _merge_environment_config(self):
        """Merge environment-specific configuration"""
        
        env_config = self.env_configs.get(self.environment)
        if env_config:
            # Add environment-specific settings
            self.config_data.setdefault('environment_settings', {})
            self.config_data['environment_settings'].update({
                'debug': env_config.debug,
                'log_level': env_config.log_level,
                'db_pool_size': env_config.db_pool_size,
                'db_max_overflow': env_config.db_max_overflow,
                'cache_ttl_seconds': env_config.cache_ttl_seconds,
                'model_cache_enabled': env_config.model_cache_enabled,
                'feature_cache_enabled': env_config.feature_cache_enabled,
                'api_rate_limit': env_config.api_rate_limit,
                'api_timeout_seconds': env_config.api_timeout_seconds
            })
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration when file is not available"""
        
        return {
            'data_sources': {
                'core_banking': {
                    'host': 'localhost',
                    'port': 5432,
                    'database': 'fraud_detection',
                    'schema': 'public'
                },
                'data_lake': {
                    'project_id': 'fraud-detection-dev',
                    'dataset': 'unified_customer_data'
                }
            },
            'feature_store': {
                'provider': 'feast',
                'online_store': {
                    'type': 'redis',
                    'connection_string': 'redis://localhost:6379'
                }
            },
            'models': {
                'hub_model': {
                    'algorithm': 'xgboost',
                    'version': 'v1.0'
                }
            },
            'inference': {
                'api': {
                    'host': '0.0.0.0',
                    'port': 8000
                },
                'thresholds': {
                    'high_risk': 0.8,
                    'medium_risk': 0.5,
                    'low_risk': 0.2
                }
            }
        }
    
    def _validate_configuration(self):
        """Validate configuration for required fields and types"""
        
        required_fields = [
            'data_sources.core_banking.host',
            'data_sources.core_banking.port',
            'feature_store.online_store.connection_string',
            'inference.api.host',
            'inference.api.port'
        ]
        
        missing_fields = []
        
        for field in required_fields:
            if not self._has_nested_key(field):
                missing_fields.append(field)
        
        if missing_fields:
            logger.error(f"Missing required configuration fields: {missing_fields}")
            raise ValueError(f"Invalid configuration: missing fields {missing_fields}")
        
        # Type validation
        self._validate_types()
        
        logger.info("Configuration validation passed")
    
    def _has_nested_key(self, key_path: str) -> bool:
        """Check if nested key exists in configuration"""
        
        keys = key_path.split('.')
        current = self.config_data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return False
        
        return True
    
    def _validate_types(self):
        """Validate configuration value types"""
        
        type_validations = {
            'data_sources.core_banking.port': int,
            'inference.api.port': int,
            'inference.thresholds.high_risk': (int, float),
            'inference.thresholds.medium_risk': (int, float),
            'inference.thresholds.low_risk': (int, float)
        }
        
        for key_path, expected_type in type_validations.items():
            value = self.get(key_path)
            if value is not None and not isinstance(value, expected_type):
                logger.warning(
                    f"Configuration value {key_path} has type {type(value).__name__}, "
                    f"expected {expected_type}"
                )
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        """
        
        keys = key_path.split('.')
        current = self.config_data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def set(self, key_path: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value
            value: Value to set
        """
        
        keys = key_path.split('.')
        current = self.config_data
        
        # Navigate to parent of target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the final value
        current[keys[-1]] = value
        
        logger.info(f"Configuration updated: {key_path} = {value}")
    
    def get_section(self, section_name: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        
        return self.get(section_name, {})
    
    def update_section(self, section_name: str, updates: Dict[str, Any]):
        """Update configuration section with new values"""
        
        current_section = self.get_section(section_name)
        current_section.update(updates)
        self.set(section_name, current_section)
    
    def get_database_config(self, source: str = "core_banking") -> Dict[str, Any]:
        """Get database configuration for specified source"""
        
        return self.get(f"data_sources.{source}", {})
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get model configuration"""
        
        if model_name == "hub_model":
            return self.get("models.hub_model", {})
        else:
            return self.get(f"models.spoke_models.{model_name}", {})
    
    def get_feature_store_config(self) -> Dict[str, Any]:
        """Get feature store configuration"""
        
        return self.get("feature_store", {})
    
    def get_inference_config(self) -> Dict[str, Any]:
        """Get inference configuration"""
        
        return self.get("inference", {})
    
    def get_environment_setting(self, setting_name: str, default: Any = None) -> Any:
        """Get environment-specific setting"""
        
        return self.get(f"environment_settings.{setting_name}", default)
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        
        return self.environment == "development"
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        
        return self.environment == "production"
    
    def get_log_level(self) -> str:
        """Get configured log level"""
        
        return self.get_environment_setting("log_level", "INFO")
    
    def is_debug_enabled(self) -> bool:
        """Check if debug mode is enabled"""
        
        return self.get_environment_setting("debug", False)
    
    def save_config(self, file_path: Optional[str] = None):
        """Save current configuration to file"""
        
        save_path = file_path or self.config_path
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'w') as file:
                yaml.dump(self.config_data, file, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
    
    def reload_config(self):
        """Reload configuration from file"""
        
        logger.info("Reloading configuration")
        self._load_configuration()
        self._validate_configuration()
        logger.info("Configuration reloaded successfully")
    
    def get_secrets(self) -> Dict[str, str]:
        """Get sensitive configuration values"""
        
        # These should be loaded from secure storage in production
        secrets = {}
        
        # Database credentials
        secrets['db_user'] = os.getenv('DB_USER', 'fraud_user')
        secrets['db_password'] = os.getenv('DB_PASSWORD', 'fraud_password')
        
        # API keys
        secrets['api_key'] = os.getenv('API_KEY', 'dev_api_key')
        secrets['jwt_secret'] = os.getenv('JWT_SECRET', 'dev_jwt_secret')
        
        # External service credentials
        secrets['gcp_credentials'] = os.getenv('GCP_CREDENTIALS_PATH')
        secrets['redis_password'] = os.getenv('REDIS_PASSWORD')
        
        return secrets
    
    def export_config(self, format: str = "yaml") -> str:
        """Export configuration in specified format"""
        
        if format.lower() == "yaml":
            return yaml.dump(self.config_data, default_flow_style=False, indent=2)
        elif format.lower() == "json":
            return json.dumps(self.config_data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration"""
        
        return {
            'environment': self.environment,
            'config_file': self.config_path,
            'debug_enabled': self.is_debug_enabled(),
            'log_level': self.get_log_level(),
            'data_sources': list(self.get_section('data_sources').keys()),
            'models_configured': list(self.get_section('models').keys()),
            'api_port': self.get('inference.api.port'),
            'feature_store_provider': self.get('feature_store.provider'),
            'timestamp': str(pd.Timestamp.now())
        }
    
    def validate_model_config(self, model_name: str) -> bool:
        """Validate model-specific configuration"""
        
        model_config = self.get_model_config(model_name)
        
        if not model_config:
            logger.error(f"No configuration found for model: {model_name}")
            return False
        
        required_fields = ['algorithm']
        
        for field in required_fields:
            if field not in model_config:
                logger.error(f"Missing required field '{field}' in {model_name} configuration")
                return False
        
        return True
    
    def __str__(self) -> str:
        """String representation of configuration"""
        
        return f"ConfigManager(environment={self.environment}, config_file={self.config_path})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        
        return (
            f"ConfigManager(environment='{self.environment}', "
            f"config_file='{self.config_path}', "
            f"sections={list(self.config_data.keys())})"
        )