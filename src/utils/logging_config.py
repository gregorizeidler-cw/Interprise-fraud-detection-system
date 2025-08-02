"""
Logging Configuration for Enterprise Fraud Detection System
Provides structured logging with correlation IDs and monitoring integration.
"""

import logging
import logging.config
import sys
import json
import traceback
from datetime import datetime
from typing import Dict, Any, Optional
import uuid
import threading
from contextvars import ContextVar


# Context variable for correlation ID
correlation_id: ContextVar[str] = ContextVar('correlation_id', default='')


class CorrelationIdFilter(logging.Filter):
    """Add correlation ID to log records"""
    
    def filter(self, record):
        record.correlation_id = correlation_id.get('')
        return True


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'correlation_id': getattr(record, 'correlation_id', ''),
            'thread_id': threading.current_thread().ident,
            'process_id': record.process
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 
                          'exc_text', 'stack_info', 'correlation_id']:
                log_entry['extra'] = log_entry.get('extra', {})
                log_entry['extra'][key] = value
        
        return json.dumps(log_entry)


class MetricsHandler(logging.Handler):
    """Custom handler to send metrics to monitoring system"""
    
    def __init__(self):
        super().__init__()
        self.error_count = 0
        self.warning_count = 0
        self.info_count = 0
    
    def emit(self, record):
        """Emit log record and update metrics"""
        
        if record.levelno >= logging.ERROR:
            self.error_count += 1
        elif record.levelno >= logging.WARNING:
            self.warning_count += 1
        elif record.levelno >= logging.INFO:
            self.info_count += 1
        
        # Send to monitoring system (Redis, Prometheus, etc.)
        self._send_to_monitoring(record)
    
    def _send_to_monitoring(self, record):
        """Send log metrics to monitoring system"""
        
        try:
            # This would integrate with your monitoring system
            # For example, sending to Redis for Prometheus to scrape
            pass
        except Exception:
            # Don't let monitoring errors break logging
            pass


class FraudDetectionLogger:
    """Custom logger wrapper for fraud detection specific logging"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.name = name
    
    def set_correlation_id(self, corr_id: Optional[str] = None):
        """Set correlation ID for request tracing"""
        
        if corr_id is None:
            corr_id = str(uuid.uuid4())
        
        correlation_id.set(corr_id)
        return corr_id
    
    def log_transaction_processing(self, transaction_id: str, customer_id: str, 
                                 product_type: str, processing_time_ms: float,
                                 fraud_score: float, action: str):
        """Log transaction processing details"""
        
        self.logger.info(
            "Transaction processed",
            extra={
                'event_type': 'transaction_processed',
                'transaction_id': transaction_id,
                'customer_id': customer_id,
                'product_type': product_type,
                'processing_time_ms': processing_time_ms,
                'fraud_score': fraud_score,
                'action': action
            }
        )
    
    def log_model_prediction(self, model_name: str, customer_id: str, 
                           prediction_score: float, feature_count: int,
                           model_version: str):
        """Log model prediction details"""
        
        self.logger.info(
            "Model prediction generated",
            extra={
                'event_type': 'model_prediction',
                'model_name': model_name,
                'customer_id': customer_id,
                'prediction_score': prediction_score,
                'feature_count': feature_count,
                'model_version': model_version
            }
        )
    
    def log_feature_computation(self, customer_id: str, feature_pillar: str,
                              feature_count: int, computation_time_ms: float,
                              cache_hit: bool = False):
        """Log feature computation details"""
        
        self.logger.debug(
            "Features computed",
            extra={
                'event_type': 'feature_computation',
                'customer_id': customer_id,
                'feature_pillar': feature_pillar,
                'feature_count': feature_count,
                'computation_time_ms': computation_time_ms,
                'cache_hit': cache_hit
            }
        )
    
    def log_high_risk_transaction(self, transaction_id: str, customer_id: str,
                                product_type: str, fraud_score: float,
                                reason_codes: list):
        """Log high-risk transaction for immediate attention"""
        
        self.logger.warning(
            "High-risk transaction detected",
            extra={
                'event_type': 'high_risk_transaction',
                'transaction_id': transaction_id,
                'customer_id': customer_id,
                'product_type': product_type,
                'fraud_score': fraud_score,
                'reason_codes': reason_codes,
                'alert_level': 'HIGH'
            }
        )
    
    def log_model_performance(self, model_name: str, metric_name: str,
                            metric_value: float, threshold: float,
                            alert_if_below: bool = True):
        """Log model performance metrics"""
        
        alert_triggered = (
            (alert_if_below and metric_value < threshold) or
            (not alert_if_below and metric_value > threshold)
        )
        
        level = logging.WARNING if alert_triggered else logging.INFO
        
        self.logger.log(
            level,
            f"Model performance metric: {metric_name}",
            extra={
                'event_type': 'model_performance',
                'model_name': model_name,
                'metric_name': metric_name,
                'metric_value': metric_value,
                'threshold': threshold,
                'alert_triggered': alert_triggered
            }
        )
    
    def log_system_error(self, error_message: str, error_type: str,
                        component: str, additional_context: Dict[str, Any] = None):
        """Log system errors with context"""
        
        extra_data = {
            'event_type': 'system_error',
            'error_type': error_type,
            'component': component
        }
        
        if additional_context:
            extra_data.update(additional_context)
        
        self.logger.error(error_message, extra=extra_data)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, extra=kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, extra=kwargs)


def setup_logging(config_manager=None, log_level: str = "INFO", 
                 log_format: str = "json", enable_file_logging: bool = True):
    """
    Setup comprehensive logging configuration.
    
    Args:
        config_manager: Configuration manager instance
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_format: Log format ('json' or 'standard')
        enable_file_logging: Whether to enable file logging
    """
    
    # Get log level from config if available
    if config_manager:
        log_level = config_manager.get_log_level()
        enable_file_logging = not config_manager.is_development()
    
    # Configure logging
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'json': {
                '()': JSONFormatter
            },
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s [%(correlation_id)s]'
            }
        },
        'filters': {
            'correlation_id': {
                '()': CorrelationIdFilter
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'stream': sys.stdout,
                'formatter': log_format,
                'filters': ['correlation_id'],
                'level': log_level
            },
            'metrics': {
                '()': MetricsHandler,
                'level': 'WARNING'
            }
        },
        'loggers': {
            '': {  # Root logger
                'handlers': ['console', 'metrics'],
                'level': log_level,
                'propagate': False
            },
            'src': {  # Application logger
                'handlers': ['console', 'metrics'],
                'level': log_level,
                'propagate': False
            },
            'uvicorn': {
                'handlers': ['console'],
                'level': 'INFO',
                'propagate': False
            },
            'sqlalchemy.engine': {
                'handlers': ['console'],
                'level': 'WARNING',
                'propagate': False
            }
        }
    }
    
    # Add file handler if enabled
    if enable_file_logging:
        logging_config['handlers']['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'logs/fraud_detection.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': log_format,
            'filters': ['correlation_id'],
            'level': log_level
        }
        
        # Add file handler to all loggers
        for logger_config in logging_config['loggers'].values():
            if 'file' not in logger_config['handlers']:
                logger_config['handlers'].append('file')
        
        # Create logs directory
        import os
        os.makedirs('logs', exist_ok=True)
    
    # Apply configuration
    logging.config.dictConfig(logging_config)
    
    # Set up custom logger instances
    setup_fraud_detection_loggers()


def setup_fraud_detection_loggers():
    """Setup fraud detection specific loggers"""
    
    # Create specialized loggers
    transaction_logger = FraudDetectionLogger('fraud_detection.transactions')
    model_logger = FraudDetectionLogger('fraud_detection.models')
    feature_logger = FraudDetectionLogger('fraud_detection.features')
    api_logger = FraudDetectionLogger('fraud_detection.api')
    
    # Store in module for easy access
    import sys
    current_module = sys.modules[__name__]
    
    current_module.transaction_logger = transaction_logger
    current_module.model_logger = model_logger
    current_module.feature_logger = feature_logger
    current_module.api_logger = api_logger


def get_logger(name: str) -> FraudDetectionLogger:
    """Get a fraud detection logger instance"""
    
    return FraudDetectionLogger(f'fraud_detection.{name}')


def log_performance_metrics(component: str, metrics: Dict[str, float]):
    """Log performance metrics for monitoring"""
    
    logger = get_logger('performance')
    
    for metric_name, metric_value in metrics.items():
        logger.info(
            f"Performance metric: {metric_name}",
            extra={
                'event_type': 'performance_metric',
                'component': component,
                'metric_name': metric_name,
                'metric_value': metric_value
            }
        )


def log_audit_event(event_type: str, user_id: str, resource: str, 
                   action: str, result: str, additional_data: Dict[str, Any] = None):
    """Log audit events for compliance"""
    
    logger = get_logger('audit')
    
    audit_data = {
        'event_type': 'audit_event',
        'audit_event_type': event_type,
        'user_id': user_id,
        'resource': resource,
        'action': action,
        'result': result,
        'timestamp': datetime.utcnow().isoformat()
    }
    
    if additional_data:
        audit_data.update(additional_data)
    
    logger.info(f"Audit event: {event_type}", extra=audit_data)


# Module-level convenience functions
def set_correlation_id(corr_id: Optional[str] = None) -> str:
    """Set correlation ID for current context"""
    
    if corr_id is None:
        corr_id = str(uuid.uuid4())
    
    correlation_id.set(corr_id)
    return corr_id


def get_correlation_id() -> str:
    """Get current correlation ID"""
    
    return correlation_id.get('')