# Fortif.ai RAG System - Quick Start Guide

## Prerequisites
- Docker installed
- Python 3.8+
- Google API Key ([Get one here](https://makersuite.google.com/app/apikey))

## Setup Instructions

### 1. Start Qdrant Vector Database
```bash
docker run -d -p 6333:6333 -p 6334:6334 --name qdrant qdrant/qdrant
```

### 2. Install Python Dependencies
```bash
pip install -r ../requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the `rag` directory:
```bash
GOOGLE_API_KEY=your_google_api_key_here
```

### 4. Run Data Ingestion
```bash
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
- `QDRANT_HOST`: Default is `http://localhost:6333`
- `QDRANT_COLLECTION_NAME`: Default is `fortif_ai_master_memory_google`
- `EMBEDDING_MODEL`: Default is `models/embedding-001`
- `VECTOR_DIMENSION`: Default is `768` (for Google's embedding model)

## Troubleshooting

### Qdrant not running?
Check if container is running:
```bash
docker ps | grep qdrant
```

Restart if needed:
```bash
docker restart qdrant
```

### API Key issues?
Verify your `.env` file exists and has the correct format:
```bash
cat .env
```

### Check Qdrant is accessible:
Visit http://localhost:6333/dashboard in your browser

## Architecture

- **Qdrant**: Vector database for storing embeddings
- **Google Generative AI**: Embedding model (`models/embedding-001`, 768 dimensions)
- **LangChain**: Text splitting and embedding integration
- **Patient Safety**: Filters sensitive data, patient-specific queries

## Notes

- Collection is created automatically on first run
- Data persists in Docker container (use volumes for production)
- Batch size default: 100 (configurable)
- Supports retry logic for API failures

