# Fortif.ai RAG System - Quick Start Guide

## Prerequisites
- Docker and Docker Compose installed
- Python 3.8+
- Google API Key ([Get one here](https://makersuite.google.com/app/apikey))

## Setup Instructions

### 1. Start Weaviate Vector Database
Navigate to the `rag` directory and start Weaviate using Docker Compose:
```bash
cd rag
docker-compose up -d
```

This will start Weaviate on:
- HTTP: http://localhost:8080
- gRPC: localhost:50051

### 2. Install Python Dependencies
From the project root:
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the project root (or `rag` directory):
```bash
GOOGLE_API_KEY=your_google_api_key_here
```

### 4. Run Data Ingestion
```bash
cd rag
python ingest.py
```
**First run:** Creates the collection automatically and ingests sample data.
**Subsequent runs:** Adds more data to the existing collection.

### 5. Test Queries
```bash
python query.py
```

## Usage

### Ingesting Data (`ingest.py`)
- Automatically creates collection if it doesn't exist
- Validates and processes patient records
- Generates embeddings using Google's model
- Batched processing for efficiency
- Includes error handling and retry logic

### Querying Data (`query.py`)
- Search patient memories by natural language questions
- Filter by patient ID
- Option to include/exclude sensitive data
- Returns relevance-scored results

## Configuration

Edit these constants in the scripts if needed:
- `WEAVIATE_HOST`: Default is `localhost`
- `WEAVIATE_PORT`: Default is `8080`
- `WEAVIATE_GRPC_PORT`: Default is `50051`
- `WEAVIATE_COLLECTION_NAME`: Default is `FortifAiMasterMemory`
- `EMBEDDING_MODEL`: Default is `models/embedding-001`
- `VECTOR_DIMENSION`: Default is `768` (for Google's embedding model)

Or set them via environment variables:
```bash
export WEAVIATE_HOST=localhost
export WEAVIATE_PORT=8080
export WEAVIATE_GRPC_PORT=50051
```

## Troubleshooting

### Weaviate not running?
Check if containers are running:
```bash
docker-compose ps
```
or
```bash
docker ps | grep weaviate
```

Restart if needed:
```bash
docker-compose restart
```

### API Key issues?
Verify your `.env` file exists and has the correct format:
```bash
cat .env
```

### Check Weaviate is accessible:
Visit http://localhost:8080/v1/meta in your browser to see Weaviate metadata

### View Weaviate logs:
```bash
docker-compose logs -f weaviate
```

## Architecture

- **Weaviate**: Vector database for storing embeddings (v1.33.3)
- **Google Generative AI**: Embedding model (`models/embedding-001`, 768 dimensions)
- **LangChain**: Text splitting and embedding integration
- **Patient Safety**: Filters sensitive data, patient-specific queries

## Notes

- Collection is created automatically on first run with proper schema
- Data persists in Docker volume `weaviate_data` (configured in docker-compose.yml)
- Batch size default: 100 (configurable)
- Supports retry logic for API failures
- Uses self-provided vectors with custom Google embeddings
- Connection cleanup handled automatically with context managers

## Stopping Weaviate

To stop the Weaviate service:
```bash
docker-compose down
```

To stop and remove all data:
```bash
docker-compose down -v
```

