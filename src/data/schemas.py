"""
Data schemas and models for the enterprise fraud detection system.
Defines the unified customer view and all data structures.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
import uuid


class ProductType(str, Enum):
    """Types of financial products"""
    CHECKING_ACCOUNT = "checking_account"
    SAVINGS_ACCOUNT = "savings_account"
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    LOAN = "loan"
    INVESTMENT = "investment"
    PIX = "pix"
    TED = "ted"
    DOC = "doc"


class TransactionType(str, Enum):
    """Types of transactions"""
    PURCHASE = "purchase"
    TRANSFER = "transfer"
    WITHDRAWAL = "withdrawal"
    DEPOSIT = "deposit"
    PAYMENT = "payment"
    INVESTMENT = "investment"
    LOAN_DISBURSEMENT = "loan_disbursement"
    LOAN_PAYMENT = "loan_payment"


class RiskLevel(str, Enum):
    """Risk classification levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DeviceType(str, Enum):
    """Device types for tracking"""
    MOBILE_APP = "mobile_app"
    WEB_BROWSER = "web_browser"
    ATM = "atm"
    POS = "pos"
    CALL_CENTER = "call_center"


# Core Customer Models

class CustomerIdentity(BaseModel):
    """Customer identity information for entity resolution"""
    customer_id: str = Field(..., description="Unique customer identifier")
    document_number: str = Field(..., description="CPF/CNPJ")
    full_name: str
    birth_date: Optional[datetime]
    email: Optional[str]
    phone: Optional[str]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True


class CustomerProfile(BaseModel):
    """Unified customer profile - 360° view"""
    customer_id: str
    identity: CustomerIdentity
    
    # Demographics
    age: Optional[int]
    gender: Optional[str]
    income_bracket: Optional[str]
    education_level: Optional[str]
    occupation: Optional[str]
    
    # Geographic
    primary_address: Optional[str]
    city: Optional[str]
    state: Optional[str]
    country: str = "BR"
    zip_code: Optional[str]
    
    # Banking Relationship
    customer_since: datetime
    relationship_length_days: int
    total_products: int
    active_products: List[ProductType]
    primary_product: Optional[ProductType]
    
    # Risk Indicators
    is_pep: bool = False  # Politically Exposed Person
    credit_score_internal: Optional[int]
    credit_score_external: Optional[int]
    has_fraud_history: bool = False
    
    # Calculated at runtime
    risk_score: Optional[float] = None
    last_activity: Optional[datetime] = None
    
    class Config:
        orm_mode = True


class Transaction(BaseModel):
    """Core transaction model"""
    transaction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    customer_id: str
    product_type: ProductType
    transaction_type: TransactionType
    
    # Transaction Details
    amount: Decimal
    currency: str = "BRL"
    description: Optional[str]
    
    # Parties
    beneficiary_id: Optional[str]
    beneficiary_document: Optional[str]
    beneficiary_name: Optional[str]
    beneficiary_bank: Optional[str]
    
    merchant_id: Optional[str]
    merchant_category: Optional[str]
    merchant_name: Optional[str]
    
    # Location & Channel
    channel: DeviceType
    device_id: Optional[str]
    ip_address: Optional[str]
    location_lat: Optional[float]
    location_lon: Optional[float]
    city: Optional[str]
    country: str = "BR"
    
    # Timing
    timestamp: datetime
    created_at: datetime
    processed_at: Optional[datetime]
    
    # Status
    status: str = "pending"  # pending, approved, rejected, manual_review
    fraud_score: Optional[float] = None
    risk_level: Optional[RiskLevel] = None
    
    class Config:
        orm_mode = True
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class CustomerEvent(BaseModel):
    """Customer behavioral events"""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    customer_id: str
    event_type: str  # login, logout, password_change, profile_update, etc.
    
    # Event Details
    timestamp: datetime
    device_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    session_id: Optional[str]
    
    # Event Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    error_code: Optional[str]
    
    class Config:
        orm_mode = True


class AccountProduct(BaseModel):
    """Customer product/account information"""
    account_id: str
    customer_id: str
    product_type: ProductType
    
    # Account Details
    account_number: str
    status: str  # active, inactive, closed, suspended
    balance: Optional[Decimal]
    credit_limit: Optional[Decimal]
    
    # Dates
    opened_date: datetime
    closed_date: Optional[datetime]
    last_activity: Optional[datetime]
    
    # Risk Indicators
    days_since_opening: int
    transaction_count_30d: int = 0
    transaction_volume_30d: Decimal = Decimal('0')
    
    class Config:
        orm_mode = True


# Feature Store Models

class FeatureVector(BaseModel):
    """Feature vector for model inference"""
    customer_id: str
    timestamp: datetime
    
    # Pillar 1: Profile Features
    profile_features: Dict[str, Any] = Field(default_factory=dict)
    
    # Pillar 2: Behavioral Features  
    behavioral_features: Dict[str, Any] = Field(default_factory=dict)
    
    # Pillar 3: Network Features
    network_features: Dict[str, Any] = Field(default_factory=dict)
    
    # Pillar 4: Contextual Features
    contextual_features: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        orm_mode = True


class ModelPrediction(BaseModel):
    """Model prediction result"""
    prediction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    customer_id: str
    transaction_id: Optional[str]
    
    # Model Information
    model_name: str
    model_version: str
    timestamp: datetime
    
    # Prediction Results
    fraud_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel
    predicted_class: str  # fraud, legitimate
    confidence: float = Field(..., ge=0.0, le=1.0)
    
    # Feature Importance (top 10)
    feature_importance: Dict[str, float] = Field(default_factory=dict)
    
    # Decision
    action: str  # approve, reject, challenge, manual_review
    reason_codes: List[str] = Field(default_factory=list)
    
    class Config:
        orm_mode = True


# Network Analysis Models

class CustomerRelationship(BaseModel):
    """Relationship between customers (network graph)"""
    source_customer_id: str
    target_customer_id: str
    relationship_type: str  # shared_device, beneficiary, frequent_transfer
    
    # Relationship Strength
    interaction_count: int
    total_amount: Decimal
    first_interaction: datetime
    last_interaction: datetime
    
    # Risk Indicators
    risk_score: float = 0.0
    is_suspicious: bool = False
    
    class Config:
        orm_mode = True


class DeviceFingerprint(BaseModel):
    """Device fingerprinting for fraud detection"""
    device_id: str
    customer_ids: List[str]  # Customers who used this device
    
    # Device Information
    device_type: DeviceType
    os: Optional[str]
    browser: Optional[str]
    screen_resolution: Optional[str]
    timezone: Optional[str]
    
    # Risk Indicators
    customer_count: int
    is_shared_device: bool = False
    risk_score: float = 0.0
    
    # Activity
    first_seen: datetime
    last_seen: datetime
    total_sessions: int = 0
    
    class Config:
        orm_mode = True


# Configuration Models

class ModelConfig(BaseModel):
    """Model configuration"""
    name: str
    version: str
    algorithm: str
    features: List[str]
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Training Configuration
    training_data_start: datetime
    training_data_end: datetime
    validation_split: float = 0.2
    
    # Performance Metrics
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]
    auc_roc: Optional[float]
    
    # Deployment
    is_active: bool = False
    deployed_at: Optional[datetime]
    
    class Config:
        orm_mode = True


class ThresholdConfig(BaseModel):
    """Risk threshold configuration"""
    model_name: str
    low_risk_threshold: float = 0.2
    medium_risk_threshold: float = 0.5
    high_risk_threshold: float = 0.8
    
    # Actions
    low_risk_action: str = "approve"
    medium_risk_action: str = "challenge"
    high_risk_action: str = "reject"
    
    updated_at: datetime
    
    class Config:
        orm_mode = True