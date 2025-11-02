#!/bin/bash
# Fortif.ai RAG System - Setup Script

echo "=== Fortif.ai RAG System Setup ==="
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi
echo "✅ Docker is running"

# Check if Qdrant container exists
if docker ps -a --format '{{.Names}}' | grep -q "^qdrant$"; then
    echo "⚠️  Qdrant container already exists"
    read -p "Do you want to restart it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker restart qdrant
        echo "✅ Qdrant container restarted"
    fi
else
    echo "🚀 Starting Qdrant container..."
    docker run -d -p 6333:6333 -p 6334:6334 --name qdrant qdrant/qdrant
    echo "✅ Qdrant container started"
    sleep 3  # Wait for Qdrant to initialize
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo ""
    echo "⚠️  .env file not found"
    read -p "Enter your Google API Key: " api_key
    echo "GOOGLE_API_KEY=$api_key" > .env
    echo "✅ .env file created"
else
    echo "✅ .env file exists"
fi

# Check if dependencies are installed
echo ""
echo "📦 Checking Python dependencies..."
if python -c "import qdrant_client" 2>/dev/null; then
    echo "✅ Dependencies appear to be installed"
else
    echo "⚠️  Installing dependencies..."
    pip install -r ../requirements.txt
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Run: python ingest.py   (to ingest sample data)"
echo "  2. Run: python query.py    (to test queries)"
echo ""
echo "Qdrant Dashboard: http://localhost:6333/dashboard"

