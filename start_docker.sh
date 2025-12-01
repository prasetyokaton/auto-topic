#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}🐳 Insights Generator - Docker Quick Start${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Function to print colored messages
success() { echo -e "${GREEN}✅ $1${NC}"; }
error() { echo -e "${RED}❌ $1${NC}"; }
warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

# Check if .env file exists
if [ ! -f ".secretcontainer/.env" ]; then
    error ".env file not found!"
    echo ""
    echo "Please create .secretcontainer/.env with:"
    echo "OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx"
    echo "OPENAI_MODEL=gpt-4o-mini"
    echo ""
    exit 1
fi

success ".env file found"

# Check Docker
if ! command -v docker &> /dev/null; then
    error "Docker is not installed!"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

success "Docker is installed"

# Check docker-compose
if ! command -v docker-compose &> /dev/null; then
    warning "docker-compose not found, using 'docker compose' instead"
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

# Stop existing container
info "Stopping existing container (if any)..."
$COMPOSE_CMD down 2>/dev/null || true

# Build
echo ""
info "Building Docker image..."
if $COMPOSE_CMD build; then
    success "Build successful"
else
    error "Build failed!"
    exit 1
fi

# Start
echo ""
info "Starting container..."
if $COMPOSE_CMD up -d; then
    success "Container started"
else
    error "Failed to start container!"
    exit 1
fi

# Wait for container to be ready
echo ""
info "Waiting for container to be ready..."
sleep 5

# Test API
echo ""
info "Testing OpenAI API connection..."
echo ""
echo -e "${BLUE}--- API Test Output ---${NC}"

if docker exec insights_app python test_api.py; then
    echo ""
    success "API test passed!"
else
    echo ""
    error "API test failed!"
    echo ""
    warning "Common issues:"
    echo "  1. Invalid API key in .env file"
    echo "  2. No internet connection"
    echo "  3. OpenAI service is down"
    echo ""
    echo "Check logs with: docker logs insights_app"
    exit 1
fi

# Show container info
echo ""
echo -e "${BLUE}============================================================${NC}"
success "Application is running!"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo "🌐 Local URL:  http://localhost:7860"
echo ""
echo "📊 Useful commands:"
echo "  View logs:     $COMPOSE_CMD logs -f"
echo "  Stop app:      $COMPOSE_CMD down"
echo "  Restart app:   $COMPOSE_CMD restart"
echo "  Test API:      docker exec insights_app python test_api.py"
echo "  Shell access:  docker exec -it insights_app bash"
echo ""
echo -e "${BLUE}============================================================${NC}"

# Show recent logs
echo ""
info "Recent logs:"
echo -e "${BLUE}--- Last 20 lines ---${NC}"
docker logs insights_app --tail 20
echo ""

echo -e "${GREEN}🎉 Ready to use!${NC}"
echo ""