"""
FastAPI REST API for Real-time Fraud Detection
Production-ready API for fraud scoring with comprehensive monitoring.
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import time
import json

from .fraud_detection_engine import (
    FraudDetectionEngine, FraudDetectionRequest, 
    FraudDetectionResponse, DecisionAction, RiskLevel
)
from ..data.schemas import ProductType
from ..utils.config_manager import ConfigManager


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize configuration and fraud detection engine
config_manager = ConfigManager()
fraud_engine = FraudDetectionEngine(config_manager)

# Create FastAPI app
app = FastAPI(
    title="Enterprise Fraud Detection API",
    description="Real-time fraud detection system with Hub and Spoke architecture",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Request/Response Models

class TransactionRequest(BaseModel):
    """Request model for fraud detection"""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    customer_id: str = Field(..., description="Customer identifier")
    product_type: ProductType = Field(..., description="Product type")
    
    # Transaction data
    amount: float = Field(..., gt=0, description="Transaction amount")
    currency: str = Field(default="BRL", description="Currency code")
    channel: str = Field(..., description="Transaction channel")
    timestamp: Optional[datetime] = Field(default=None, description="Transaction timestamp")
    
    # Optional context
    device_id: Optional[str] = Field(None, description="Device identifier")
    ip_address: Optional[str] = Field(None, description="IP address")
    location_lat: Optional[float] = Field(None, description="Latitude")
    location_lon: Optional[float] = Field(None, description="Longitude")
    
    # Product-specific fields
    beneficiary_id: Optional[str] = Field(None, description="Beneficiary identifier for transfers")
    beneficiary_bank: Optional[str] = Field(None, description="Beneficiary bank")
    merchant_id: Optional[str] = Field(None, description="Merchant identifier")
    merchant_category: Optional[str] = Field(None, description="Merchant category code")
    
    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "txn_12345",
                "customer_id": "cust_67890", 
                "product_type": "pix",
                "amount": 1500.00,
                "currency": "BRL",
                "channel": "mobile_app",
                "device_id": "device_abc123",
                "ip_address": "192.168.1.1",
                "beneficiary_id": "beneficiary_999"
            }
        }
    
    @validator('timestamp', pre=True, always=True)
    def set_timestamp(cls, v):
        return v or datetime.now()


class FraudDetectionResponseModel(BaseModel):
    """Response model for fraud detection"""
    transaction_id: str
    customer_id: str
    product_type: str
    
    # Scores
    hub_risk_score: float = Field(..., ge=0, le=1)
    spoke_fraud_score: float = Field(..., ge=0, le=1)  
    final_score: float = Field(..., ge=0, le=1)
    
    # Classification
    risk_level: RiskLevel
    predicted_class: str
    confidence: float = Field(..., ge=0, le=1)
    
    # Decision
    action: DecisionAction
    reason_codes: List[str]
    
    # Metadata
    processing_time_ms: float
    model_versions: Dict[str, str]
    timestamp: datetime
    
    # Optional explanation (can be disabled for performance)
    explanation: Optional[Dict[str, Any]] = None


class BatchFraudRequest(BaseModel):
    """Request model for batch fraud detection"""
    transactions: List[TransactionRequest] = Field(..., max_items=100)
    include_explanation: bool = Field(default=False)


class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: datetime
    components: Dict[str, Any]
    performance_metrics: Optional[Dict[str, Any]] = None


class MetricsResponse(BaseModel):
    """Metrics response model"""
    total_requests: int
    average_processing_time_ms: float
    requests_per_second: float
    timestamp: datetime


# Middleware for request logging and timing

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log incoming request
    logger.info(f"Incoming request: {request.method} {request.url}")
    
    # Process request
    response = await call_next(request)
    
    # Log response time
    process_time = time.time() - start_time
    logger.info(f"Request completed in {process_time*1000:.2f}ms")
    
    # Add timing header
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


# Exception handlers

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={
            "error": "Bad Request",
            "message": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )


# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "service": "Enterprise Fraud Detection API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/fraud-detection", response_model=FraudDetectionResponseModel)
async def detect_fraud(
    request: TransactionRequest,
    include_explanation: bool = False,
    background_tasks: BackgroundTasks = None
):
    """
    Detect fraud for a single transaction.
    
    This is the main endpoint for real-time fraud detection.
    Returns fraud score, risk level, and recommended action.
    """
    
    try:
        # Convert request to internal format
        transaction_data = {
            "amount": request.amount,
            "currency": request.currency,
            "channel": request.channel,
            "timestamp": request.timestamp.isoformat(),
            "device_id": request.device_id,
            "ip_address": request.ip_address,
            "location_lat": request.location_lat,
            "location_lon": request.location_lon,
            "beneficiary_id": request.beneficiary_id,
            "beneficiary_bank": request.beneficiary_bank,
            "merchant_id": request.merchant_id,
            "merchant_category": request.merchant_category
        }
        
        fraud_request = FraudDetectionRequest(
            transaction_id=request.transaction_id,
            customer_id=request.customer_id,
            product_type=request.product_type,
            transaction_data=transaction_data,
            timestamp=request.timestamp
        )
        
        # Perform fraud detection
        result = await fraud_engine.detect_fraud(fraud_request)
        
        # Convert to response format
        response = FraudDetectionResponseModel(
            transaction_id=result.transaction_id,
            customer_id=result.customer_id,
            product_type=result.product_type.value,
            hub_risk_score=result.hub_risk_score,
            spoke_fraud_score=result.spoke_fraud_score,
            final_score=result.final_score,
            risk_level=result.risk_level,
            predicted_class=result.predicted_class,
            confidence=result.confidence,
            action=result.action,
            reason_codes=result.reason_codes,
            processing_time_ms=result.processing_time_ms,
            model_versions=result.model_versions,
            timestamp=result.timestamp,
            explanation=result.explanation if include_explanation else None
        )
        
        # Log high-risk transactions in background
        if result.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            background_tasks.add_task(log_high_risk_transaction, result)
        
        return response
        
    except Exception as e:
        logger.error(f"Error in fraud detection: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Fraud detection failed: {str(e)}"
        )


@app.post("/fraud-detection/batch", response_model=List[FraudDetectionResponseModel])
async def detect_fraud_batch(request: BatchFraudRequest):
    """
    Batch fraud detection for multiple transactions.
    
    Processes up to 100 transactions in parallel.
    Useful for batch processing scenarios.
    """
    
    if len(request.transactions) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 transactions per batch request"
        )
    
    try:
        # Convert all requests to internal format
        fraud_requests = []
        for txn in request.transactions:
            transaction_data = {
                "amount": txn.amount,
                "currency": txn.currency,
                "channel": txn.channel,
                "timestamp": txn.timestamp.isoformat(),
                "device_id": txn.device_id,
                "ip_address": txn.ip_address,
                "location_lat": txn.location_lat,
                "location_lon": txn.location_lon,
                "beneficiary_id": txn.beneficiary_id,
                "beneficiary_bank": txn.beneficiary_bank,
                "merchant_id": txn.merchant_id,
                "merchant_category": txn.merchant_category
            }
            
            fraud_request = FraudDetectionRequest(
                transaction_id=txn.transaction_id,
                customer_id=txn.customer_id,
                product_type=txn.product_type,
                transaction_data=transaction_data,
                timestamp=txn.timestamp
            )
            fraud_requests.append(fraud_request)
        
        # Process all requests in parallel
        tasks = [fraud_engine.detect_fraud(req) for req in fraud_requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert successful results to response format
        responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing transaction {fraud_requests[i].transaction_id}: {result}")
                # Create error response
                error_response = FraudDetectionResponseModel(
                    transaction_id=fraud_requests[i].transaction_id,
                    customer_id=fraud_requests[i].customer_id,
                    product_type=fraud_requests[i].product_type.value,
                    hub_risk_score=0.5,
                    spoke_fraud_score=0.5,
                    final_score=0.5,
                    risk_level=RiskLevel.MEDIUM,
                    predicted_class="error",
                    confidence=0.0,
                    action=DecisionAction.MANUAL_REVIEW,
                    reason_codes=["PROCESSING_ERROR"],
                    processing_time_ms=0.0,
                    model_versions={"error": str(result)},
                    timestamp=datetime.now()
                )
                responses.append(error_response)
            else:
                response = FraudDetectionResponseModel(
                    transaction_id=result.transaction_id,
                    customer_id=result.customer_id,
                    product_type=result.product_type.value,
                    hub_risk_score=result.hub_risk_score,
                    spoke_fraud_score=result.spoke_fraud_score,
                    final_score=result.final_score,
                    risk_level=result.risk_level,
                    predicted_class=result.predicted_class,
                    confidence=result.confidence,
                    action=result.action,
                    reason_codes=result.reason_codes,
                    processing_time_ms=result.processing_time_ms,
                    model_versions=result.model_versions,
                    timestamp=result.timestamp,
                    explanation=result.explanation if request.include_explanation else None
                )
                responses.append(response)
        
        return responses
        
    except Exception as e:
        logger.error(f"Error in batch fraud detection: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch fraud detection failed: {str(e)}"
        )


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns the health status of the fraud detection system
    including all dependencies and components.
    """
    
    try:
        # Get health status from engine
        health_status = await fraud_engine.health_check()
        
        # Add performance metrics
        performance_metrics = fraud_engine.get_performance_metrics()
        
        return HealthCheckResponse(
            status=health_status['status'],
            timestamp=datetime.fromisoformat(health_status['timestamp']),
            components=health_status['components'],
            performance_metrics=performance_metrics
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            timestamp=datetime.now(),
            components={"error": str(e)}
        )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get performance metrics.
    
    Returns current performance statistics including
    request counts, processing times, and throughput.
    """
    
    try:
        metrics = fraud_engine.get_performance_metrics()
        
        return MetricsResponse(
            total_requests=metrics['total_requests'],
            average_processing_time_ms=metrics['average_processing_time_ms'],
            requests_per_second=metrics['requests_per_second'],
            timestamp=datetime.fromisoformat(metrics['timestamp'])
        )
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get metrics: {str(e)}"
        )


@app.get("/models/info")
async def get_model_info():
    """
    Get information about loaded models.
    
    Returns metadata about the Hub and Spoke models
    including versions, performance metrics, and status.
    """
    
    try:
        # Get hub model info
        hub_info = {}
        if fraud_engine.hub_model_manager.model and fraud_engine.hub_model_manager.model.is_trained:
            hub_info = {
                "status": "loaded",
                "metadata": fraud_engine.hub_model_manager.model_metadata,
                "feature_importance": dict(
                    sorted(
                        fraud_engine.hub_model_manager.model.feature_importance.items(),
                        key=lambda x: x[1], reverse=True
                    )[:10]  # Top 10 features
                )
            }
        else:
            hub_info = {"status": "not_loaded"}
        
        # Get spoke models info
        spoke_info = fraud_engine.spoke_model_manager.get_model_performance()
        
        return {
            "hub_model": hub_info,
            "spoke_models": spoke_info,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        )


@app.post("/models/reload")
async def reload_models():
    """
    Reload all models from disk.
    
    Useful for deploying new model versions without
    restarting the entire service.
    """
    
    try:
        # Reload models
        fraud_engine._load_models()
        
        return {
            "status": "success",
            "message": "Models reloaded successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error reloading models: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload models: {str(e)}"
        )


# Background tasks

async def log_high_risk_transaction(result: FraudDetectionResponse):
    """Log high-risk transactions for immediate attention"""
    
    try:
        logger.warning(
            f"HIGH RISK TRANSACTION: {result.transaction_id} "
            f"Customer: {result.customer_id} "
            f"Score: {result.final_score:.3f} "
            f"Action: {result.action.value}"
        )
        
        # Could send alerts, notifications, etc.
        
    except Exception as e:
        logger.error(f"Error logging high-risk transaction: {e}")


# Startup and shutdown events

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    
    logger.info("Starting Enterprise Fraud Detection API")
    
    try:
        # Perform health check
        health = await fraud_engine.health_check()
        
        if health['status'] == 'healthy':
            logger.info("Fraud detection engine is healthy")
        else:
            logger.warning(f"Fraud detection engine health issues: {health}")
        
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources"""
    
    logger.info("Shutting down Enterprise Fraud Detection API")
    
    try:
        # Close database connections, Redis connections, etc.
        # This would be implemented based on the specific infrastructure
        logger.info("Cleanup completed successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Run the application
if __name__ == "__main__":
    # Configuration
    host = config_manager.get("inference.api.host", "0.0.0.0")
    port = config_manager.get("inference.api.port", 8000)
    workers = config_manager.get("inference.api.max_workers", 1)
    
    # Start the server
    uvicorn.run(
        "src.inference.api:app",
        host=host,
        port=port,
        workers=workers,
        reload=False,  # Set to True for development
        access_log=True
    )