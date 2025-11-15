#!/bin/bash
# Fortif.ai RAG System - Setup Script

echo "=== Fortif.ai RAG System Setup (Weaviate) ==="
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi
echo "✅ Docker is running"

# --- NEW: Check and start Weaviate container ---
WEAVIATE_NAME="weaviate"
if docker ps -a --format '{{.Names}}' | grep -q "^${WEAVIATE_NAME}$"; then
    echo "⚠️  Weaviate container already exists"
    read -p "Do you want to restart it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker restart ${WEAVIATE_NAME}
        echo "✅ Weaviate container restarted"
    fi
else
    echo "🚀 Starting Weaviate container..." 
    # Weaviate requires a vectorizer (like text2vec-none if we provide our own vectors)
    docker pull semitechnologies/weaviate:1.23.9
    docker run -d \
        -p 8080:8080 \
        --name ${WEAVIATE_NAME} \
        -e "QUERY_DEFAULTS_LIMIT=100" \
        -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
        -e "PERSISTENCE_DATA_PATH=/var/lib/weaviate" \
        -e ENABLE_MODULES=text2vec-openai \
        semitechnologies/weaviate:1.23.9
    echo "✅ Weaviate container started"
    sleep 5  # Wait for Weaviate to initialize
fi

# Set the WEAVIATE_URL environment variable (user must manually add to .env later)
WEAVIATE_URL_DEFAULT="http://localhost:8080"
echo "Weaviate default URL set to: ${WEAVIATE_URL_DEFAULT}"

# Check if .env file exists
if [ ! -f .env ]; then
    echo ""
    echo "⚠️  .env file not found"
    read -p "Enter your Google API Key: " api_key
    echo "GOOGLE_API_KEY=$api_key" > .env
    echo "WEAVIATE_URL=${WEAVIATE_URL_DEFAULT}" >> .env
    echo "✅ .env file created"
else
    echo "✅ .env file exists"
    # Ensure WEAVIATE_URL is in .env
    if ! grep -q "^WEAVIATE_URL=" .env; then
        echo "WEAVIATE_URL=${WEAVIATE_URL_DEFAULT}" >> .env
        echo "✅ WEAVIATE_URL added to .env"
    fi
fi

# Check if dependencies are installed
echo ""
python3 -m venv ../venv
source ../venv/bin/activate
echo "📦 Checking Python dependencies..."
# Check for Weaviate client now
if python -c "import weaviate" 2>/dev/null; then
    echo "✅ Dependencies appear to be installed"
else
    echo "⚠️  Installing dependencies (Weaviate, etc.)..."
    pip install -r ../requirements.txt
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Run: python ingest.py   (to create schema and ingest sample data)"
echo "  2. Run: python query.py    (to test queries)"
echo ""
echo "Weaviate Dashboard: ${WEAVIATE_URL_DEFAULT}/v1/graphql"