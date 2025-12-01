#!/bin/bash

# Simple Docker Test Commands for Insights Generator
# Usage: ./test_docker.sh [command]

CONTAINER_NAME="insights_app"

case "$1" in
  "api")
    echo "🧪 Testing OpenAI API connection..."
    docker exec $CONTAINER_NAME python test_api.py
    ;;
  
  "env")
    echo "🔍 Checking environment variables..."
    docker exec $CONTAINER_NAME printenv | grep OPENAI
    ;;
  
  "files")
    echo "📁 Listing files in container..."
    docker exec $CONTAINER_NAME ls -la /app/
    ;;
  
  "logs")
    echo "📊 Showing container logs..."
    docker logs $CONTAINER_NAME --tail 50
    ;;
  
  "health")
    echo "❤️  Checking container health..."
    docker inspect --format='{{.State.Health.Status}}' $CONTAINER_NAME 2>/dev/null || echo "No health check configured"
    ;;
  
  "curl")
    echo "🌐 Testing Gradio endpoint..."
    docker exec $CONTAINER_NAME curl -s http://localhost:7860 | head -20
    ;;
  
  "shell")
    echo "🐚 Opening shell in container..."
    docker exec -it $CONTAINER_NAME bash
    ;;
  
  "restart")
    echo "🔄 Restarting container..."
    docker restart $CONTAINER_NAME
    ;;
  
  "clean")
    echo "🧹 Cleaning up..."
    docker-compose down
    docker system prune -f
    echo "✅ Cleanup complete"
    ;;
  
  "rebuild")
    echo "🔨 Rebuilding and restarting..."
    docker-compose down
    docker-compose build --no-cache
    docker-compose up -d
    sleep 5
    docker exec $CONTAINER_NAME python test_api.py
    ;;
  
  *)
    echo "Usage: ./test_docker.sh [command]"
    echo ""
    echo "Available commands:"
    echo "  api      - Test OpenAI API connection"
    echo "  env      - Check environment variables"
    echo "  files    - List files in container"
    echo "  logs     - Show container logs"
    echo "  health   - Check container health status"
    echo "  curl     - Test Gradio endpoint"
    echo "  shell    - Open bash shell in container"
    echo "  restart  - Restart container"
    echo "  clean    - Clean up containers and images"
    echo "  rebuild  - Rebuild from scratch"
    echo ""
    echo "Examples:"
    echo "  ./test_docker.sh api"
    echo "  ./test_docker.sh logs"
    echo "  ./test_docker.sh shell"
    ;;
esac