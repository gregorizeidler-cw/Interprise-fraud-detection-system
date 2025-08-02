#!/bin/bash

# Enterprise Fraud Detection System - Startup Script
# This script starts the complete fraud detection system

set -e  # Exit on any error

echo "🚀 Starting Enterprise Fraud Detection System"
echo "=" * 50

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose not found. Please install docker-compose."
    exit 1
fi

echo "✅ Docker is running"

# Set environment variables
export ENVIRONMENT=production
export DB_USER=fraud_user
export DB_PASSWORD=fraud_password
export DB_NAME=fraud_detection

echo "📋 Environment Configuration:"
echo "   Environment: $ENVIRONMENT"
echo "   Database: $DB_NAME"
echo "   User: $DB_USER"

# Start the system with docker-compose
echo ""
echo "🔄 Starting system components..."

# Start core services (database, cache, API)
docker-compose up -d postgres redis fraud-detection-api

echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service health
echo "🔍 Checking service health..."

# Check PostgreSQL
if docker-compose exec postgres pg_isready -U fraud_user -d fraud_detection >/dev/null 2>&1; then
    echo "✅ PostgreSQL is ready"
else
    echo "❌ PostgreSQL is not ready"
    exit 1
fi

# Check Redis
if docker-compose exec redis redis-cli ping >/dev/null 2>&1; then
    echo "✅ Redis is ready"
else
    echo "❌ Redis is not ready"
    exit 1
fi

# Check API health
echo "🔍 Checking API health..."
sleep 10

if curl -f http://localhost:8000/health >/dev/null 2>&1; then
    echo "✅ Fraud Detection API is ready"
else
    echo "❌ Fraud Detection API is not ready"
    echo "💡 Check logs: docker-compose logs fraud-detection-api"
fi

echo ""
echo "🎉 System startup completed!"
echo ""
echo "📍 Available endpoints:"
echo "   API: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo "   Health Check: http://localhost:8000/health"
echo ""
echo "🔧 Management commands:"
echo "   Check logs: docker-compose logs -f"
echo "   Stop system: docker-compose down"
echo "   Restart: docker-compose restart"
echo ""
echo "📊 Optional monitoring services:"
echo "   Start monitoring: docker-compose --profile monitoring up -d"
echo "   Grafana: http://localhost:3000 (admin/admin)"
echo "   Prometheus: http://localhost:9090"
echo ""
echo "🧪 Test the system:"
echo "   Run notebooks: jupyter lab notebooks/"
echo "   API test: curl -X POST http://localhost:8000/fraud-detection \\"
echo "             -H 'Content-Type: application/json' \\"
echo "             -d '{\"transaction_id\":\"test_001\",\"customer_id\":\"cust_123\",\"product_type\":\"pix\",\"amount\":100}'"
echo ""
echo "✨ Enterprise Fraud Detection System is now running!"