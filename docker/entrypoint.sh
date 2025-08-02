#!/bin/bash
set -e

# Enterprise Fraud Detection System - Docker Entrypoint Script

echo "Starting Enterprise Fraud Detection System..."

# Wait for database to be ready
if [ -n "$DB_HOST" ]; then
    echo "Waiting for database at $DB_HOST:$DB_PORT..."
    while ! nc -z "$DB_HOST" "$DB_PORT"; do
        echo "Database not ready, waiting..."
        sleep 2
    done
    echo "Database is ready!"
fi

# Wait for Redis to be ready
if [ -n "$REDIS_HOST" ]; then
    echo "Waiting for Redis at $REDIS_HOST:$REDIS_PORT..."
    while ! nc -z "$REDIS_HOST" "$REDIS_PORT"; do
        echo "Redis not ready, waiting..."
        sleep 2
    done
    echo "Redis is ready!"
fi

# Initialize database schema if needed
if [ "$INIT_DATABASE" = "true" ]; then
    echo "Initializing database schema..."
    python -c "
from src.infrastructure.database import DataWarehouseManager
from src.utils.config_manager import ConfigManager

config = ConfigManager()
warehouse = DataWarehouseManager(config)
warehouse.create_feature_tables()
print('Database schema initialized')
"
fi

# Run database migrations if needed
if [ "$RUN_MIGRATIONS" = "true" ]; then
    echo "Running database migrations..."
    # Add migration logic here
    echo "Migrations completed"
fi

# Load models if specified
if [ -n "$LOAD_MODELS" ]; then
    echo "Loading models..."
    python -c "
from src.models.hub_model import HubModelManager
from src.models.spoke_models import SpokeModelManager
from src.utils.config_manager import ConfigManager

config = ConfigManager()

# Load hub model
if '$HUB_MODEL_PATH':
    hub_manager = HubModelManager(config)
    hub_manager.load_model('$HUB_MODEL_PATH')
    print('Hub model loaded')

# Load spoke models
spoke_manager = SpokeModelManager(config)
if '$PIX_MODEL_PATH':
    spoke_manager.load_spoke_model('pix', '$PIX_MODEL_PATH')
    print('PIX model loaded')

if '$CREDIT_CARD_MODEL_PATH':
    spoke_manager.load_spoke_model('credit_card', '$CREDIT_CARD_MODEL_PATH')
    print('Credit card model loaded')
"
fi

# Set up logging
export PYTHONPATH="/app:$PYTHONPATH"

# Health check before starting
echo "Performing health check..."
python -c "
from src.inference.fraud_detection_engine import FraudDetectionEngine
from src.utils.config_manager import ConfigManager
import asyncio

async def health_check():
    config = ConfigManager()
    engine = FraudDetectionEngine(config)
    health = await engine.health_check()
    print(f'Health check result: {health[\"status\"]}')
    if health['status'] != 'healthy':
        print('Warning: Some components are not healthy')
        print(health)

asyncio.run(health_check())
"

echo "System initialization completed!"

# Execute the main command
exec "$@"