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

- **Weaviate**: Vector database for storing embeddings (v1.34.0)
- **Google Generative AI**: Embedding model (`models/gemini-embedding-001`, 768 dimensions)
- **LangChain**: Text splitting and embedding integration
- **Patient Safety**: Filters sensitive data, patient-specific queries
- **Performance**: gRPC enabled for 60-80% faster batch imports

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

---

# Fortif.ai RAG API Server (Phase 2)

## Quick Start - API Server

### 1. Setup Environment
```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your Google API key
nano .env  # or use your preferred editor

# (Optional) Generate API key for authentication
python -c "import secrets; print(secrets.token_urlsafe(32))"
# Add generated key to .env as API_KEY=<your_key>
```

### 2. Start Weaviate
```bash
docker-compose up -d
```

### 3. Install Dependencies
```bash
pip install -r ../requirements.txt
```

### 4. Run the API Server
```bash
# Development mode (auto-reload)
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 5. Access API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

---

## API Endpoints

### Health Check
**GET** `/api/health`

Check if the API and Weaviate are operational.

**Response:**
```json
{
  "status": "healthy",
  "weaviate_connected": true,
  "collection_exists": true,
  "collection_count": 5,
  "timestamp": "2025-11-13T10:30:00"
}
```

**Example:**
```bash
curl http://localhost:8000/api/health
```

---

### Query Patient Memories (RAG)
**POST** `/api/query`

Full RAG pipeline: Retrieve patient memories and generate empathetic response.

**Authentication:** Requires `X-API-Key` header (if API_KEY is set in .env)

**Request Body:**
```json
{
  "patient_id": "patient_123",
  "question": "Tell me about my granddaughter's birthday",
  "include_sensitive": false,
  "emotion_filter": "positive",
  "limit": 3
}
```

**Response:**
```json
{
  "response": "What a wonderful memory! Your granddaughter Sarah's 5th birthday at the park by the old oak tree sounds like such a special day. That red bicycle must have brought her so much joy! Her laughter truly is the best sound, isn't it?",
  "sources": [
    {
      "text": "A cherished memory for Jane is her granddaughter Sarah's 5th birthday party...",
      "topic": "positive_memory",
      "emotion": "positive",
      "source": "family_questionnaire",
      "distance": 0.12,
      "chunk_index": 0,
      "total_chunks": 1
    }
  ],
  "patient_id": "patient_123",
  "metadata": {
    "retrieved_count": 1,
    "model": "gemini-2.5-flash",
    "temperature": 0.3,
    "timestamp": "2025-11-13T10:35:00"
  }
}
```

**Example with curl:**
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
  -d '{
    "patient_id": "patient_123",
    "question": "Tell me about my morning routine",
    "limit": 3
  }'
```

**Example with Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/query",
    headers={"X-API-Key": "your_api_key_here"},
    json={
        "patient_id": "patient_123",
        "question": "Tell me about my morning routine",
        "limit": 3
    }
)

result = response.json()
print(result["response"])
```

**Query Parameters:**
- `patient_id` (required): Unique patient identifier
- `question` (required): Patient's question (1-500 chars)
- `include_sensitive` (optional, default: false): Include sensitive memories
- `emotion_filter` (optional): Filter by emotion (positive/negative/neutral/mixed/unknown)
- `limit` (optional, default: 3): Max documents to retrieve (1-10)

---

### Ingest Patient Memory
**POST** `/api/ingest`

Add new patient memory to the system.

**Authentication:** Requires `X-API-Key` header

**Request Body:**
```json
{
  "patient_id": "patient_123",
  "raw_text": "Jane enjoys her morning tea ritual. Every day at 7 AM, she sits by the window with her favorite blue teacup and watches the birds in the garden.",
  "source": "family_questionnaire",
  "topic": "daily_routine",
  "is_sensitive": false,
  "entities": ["tea", "garden", "birds", "blue teacup"],
  "emotion": "positive"
}
```

**Response:**
```json
{
  "status": "success",
  "patient_id": "patient_123",
  "chunks_created": 1,
  "objects_upserted": 1,
  "message": "Successfully ingested memory for patient patient_123"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/ingest \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
  -d '{
    "patient_id": "patient_456",
    "raw_text": "Bill loves telling stories about his time in the army...",
    "source": "family_interview",
    "topic": "life_history",
    "is_sensitive": false,
    "entities": ["army", "military service"],
    "emotion": "positive"
  }'
```

---

## Authentication

The API uses API Key authentication via the `X-API-Key` header.

### Development Mode (No Auth)
Leave `API_KEY` empty in `.env`:
```bash
API_KEY=
```
All requests will be accepted without authentication.

### Production Mode (With Auth)
1. Generate a secure API key:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

2. Add to `.env`:
```bash
API_KEY=your_generated_key_here
```

3. Include in all requests:
```bash
curl -H "X-API-Key: your_generated_key_here" ...
```

---

## Configuration

All configuration is managed through environment variables in `.env`:

### Key Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | *(required)* | Google AI API key |
| `API_KEY` | *(none)* | API authentication key |
| `LLM_MODEL` | `gemini-2.5-flash` | LLM for generation |
| `LLM_TEMPERATURE` | `0.3` | Response consistency (0-1) |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Embedding model |
| `VECTOR_DIMENSION` | `768` | Embedding dimensions |
| `DEFAULT_RETRIEVAL_LIMIT` | `3` | Default docs to retrieve |
| `CORS_ORIGINS` | `localhost:3000,...` | Allowed frontend origins |

See `.env.example` for full configuration options.

---

## Architecture - RAG Pipeline

```
┌─────────────┐
│   Request   │
│ (Question)  │
└──────┬──────┘
       │
       v
┌─────────────────────────────────┐
│  1. RETRIEVAL                   │
│  - Generate query embedding     │
│  - Apply filters (patient_id,   │
│    is_sensitive, emotion)       │
│  - Vector similarity search     │
│  - Return top-k documents       │
└──────┬──────────────────────────┘
       │
       v
┌─────────────────────────────────┐
│  2. AUGMENTATION                │
│  - Format retrieved docs        │
│  - Build context string         │
│  - Include metadata (topic,     │
│    emotion, source)             │
└──────┬──────────────────────────┘
       │
       v
┌─────────────────────────────────┐
│  3. GENERATION                  │
│  - Apply healthcare prompt      │
│  - Send context + question      │
│    to Gemini LLM                │
│  - Generate empathetic response │
└──────┬──────────────────────────┘
       │
       v
┌─────────────┐
│  Response   │
│ (Empathetic │
│  + Sources) │
└─────────────┘
```

### Components

- **Weaviate v1.34.0**: Vector database
- **Google Gemini 2.5 Flash**: LLM for generation (temp=0.3 for consistency)
- **Gemini Embedding 001**: Text embeddings (768 dimensions)
- **LangChain**: Prompt templates and chain orchestration
- **FastAPI**: REST API framework
- **Pydantic**: Request/response validation

---

## Prompt Engineering

The system uses healthcare-specific prompts designed for dementia care:

### Key Principles
1. **Context-Only Responses**: Never invent information
2. **Empathetic Tone**: Warm, validating, patient-focused
3. **Simple Language**: Short sentences, concrete details
4. **Memory Validation**: Never contradict cherished memories
5. **Safety Guardrails**: No medical advice, deflect sensitive topics appropriately

### Prompt Template Structure
```
You are Fortif.ai, a compassionate AI companion...

CRITICAL RULES:
1. Only use provided context
2. Never mention you're an AI
3. Validate patient memories
...

CONTEXT: {context}
QUESTION: {question}

Respond with warmth and empathy:
```

See `prompts.py` for full templates.

---

## Testing

### Manual Testing with curl

**Health Check:**
```bash
curl http://localhost:8000/api/health
```

**Query without auth (dev mode):**
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"patient_id": "patient_123", "question": "Tell me about Sarah"}'
```

**Query with auth:**
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_key" \
  -d '{"patient_id": "patient_123", "question": "Tell me about my family"}'
```

### Testing with Swagger UI

1. Navigate to http://localhost:8000/docs
2. Click "Authorize" button (if API_KEY is set)
3. Enter your API key
4. Try the `/api/query` endpoint with the "Try it out" button

### Python Testing Script

```python
import requests

BASE_URL = "http://localhost:8000"
API_KEY = "your_api_key_here"  # or None for dev mode

headers = {"X-API-Key": API_KEY} if API_KEY else {}

# Test query
response = requests.post(
    f"{BASE_URL}/api/query",
    headers=headers,
    json={
        "patient_id": "patient_123",
        "question": "What do you remember about your granddaughter?",
        "limit": 3
    }
)

print(f"Status: {response.status_code}")
print(f"Response: {response.json()['response']}")
print(f"Sources: {len(response.json()['sources'])}")
```

---

## Troubleshooting

### API won't start

**Check Weaviate is running:**
```bash
docker-compose ps
curl http://localhost:8080/v1/meta
```

**Check environment variables:**
```bash
cat .env | grep GOOGLE_API_KEY
```

**Check logs:**
```bash
# If running with uvicorn
# Logs appear in terminal

# If running as service, check process logs
```

### "Missing API key" error

Either:
1. Add `X-API-Key` header to request, OR
2. Remove `API_KEY` from `.env` for dev mode

### "Vector dimension mismatch" error

Ensure `VECTOR_DIMENSION=768` in `.env` matches the embedding model output.

### LLM generates irrelevant responses

- Check retrieved documents are relevant (inspect `sources` in response)
- Adjust `limit` parameter (try 5-7 for more context)
- Verify `LLM_TEMPERATURE` is set to 0.3 (not higher)

### Slow response times

- Use `gemini-2.5-flash` (not `gemini-2.5-pro`)
- Reduce `limit` parameter
- Check Weaviate is using gRPC (port 50051)

---

## Production Deployment

### Security Checklist
- [ ] Set strong `API_KEY` in `.env`
- [ ] Update `CORS_ORIGINS` with your frontend domain
- [ ] Use HTTPS (reverse proxy with nginx/traefik)
- [ ] Enable rate limiting (add slowapi middleware)
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Regular Weaviate backups
- [ ] Log patient queries for HIPAA compliance (if applicable)

### Performance Tuning
- Use multiple uvicorn workers: `--workers 4`
- Enable Weaviate replication for high availability
- Consider caching frequent queries (Redis)
- Monitor Google AI API usage and rate limits

---

## File Structure

```
rag/
├── main.py              # FastAPI server entry point
├── config.py            # Environment configuration
├── models.py            # Pydantic request/response schemas
├── prompts.py           # Healthcare prompt templates
├── auth.py              # API key authentication
├── rag_pipeline.py      # Core RAG logic (retrieve/augment/generate)
├── ingest.py            # Data ingestion (unchanged)
├── query.py             # CLI query tool (legacy, optional)
├── docker-compose.yml   # Weaviate setup
├── .env                 # Environment variables (create from .env.example)
├── .env.example         # Environment template
└── README.md            # This file
```

---

## Next Steps

1. **Ingest Data**: Run `python ingest.py` to add sample patient memories
2. **Start API**: Run `uvicorn main:app --reload`
3. **Test API**: Visit http://localhost:8000/docs
4. **Integrate Frontend**: Connect your UI to `/api/query` endpoint
5. **Deploy**: Follow production checklist above

For questions or issues, see the [Troubleshooting](#troubleshooting) section.

