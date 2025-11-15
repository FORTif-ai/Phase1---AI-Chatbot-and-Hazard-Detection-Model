"""
Fortif.ai RAG API Server
FastAPI application providing empathetic chatbot responses for dementia patients.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import weaviate
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from models import (
    QueryRequest,
    QueryResponse,
    SourceDocument,
    IngestRequest,
    IngestResponse,
    HealthResponse,
    ErrorResponse
)
from auth import verify_api_key
from rag_pipeline import RAGPipeline
from ingest import (
    validate_patient_record,
    ensure_collection_exists,
    process_and_ingest_data
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifespan: initialize clients on startup, cleanup on shutdown.
    """
    logger.info("=== Initializing Fortif.ai RAG Server ===")

    try:
        # Initialize Weaviate client
        logger.info(f"Connecting to Weaviate at {settings.weaviate_host}:{settings.weaviate_port}")
        app.state.weaviate_client = weaviate.connect_to_local(
            host=settings.weaviate_host,
            port=settings.weaviate_port,
            grpc_port=settings.weaviate_grpc_port
        )
        logger.info("✓ Weaviate client connected")

        # Ensure collection exists
        ensure_collection_exists(app.state.weaviate_client, settings.weaviate_collection_name)
        logger.info(f"✓ Collection '{settings.weaviate_collection_name}' ready")

        # Initialize Google embeddings for queries
        logger.info(f"Initializing Google embeddings: {settings.embedding_model}")
        app.state.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            task_type="retrieval_query"
        )
        logger.info("✓ Embedding model initialized")

        # Initialize Google LLM
        logger.info(f"Initializing Google LLM: {settings.llm_model} (temp={settings.llm_temperature})")
        app.state.llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature
        )
        logger.info("✓ LLM initialized")

        # Initialize RAG pipeline
        app.state.rag_pipeline = RAGPipeline(
            weaviate_client=app.state.weaviate_client,
            embeddings=app.state.embeddings,
            llm=app.state.llm
        )
        logger.info("✓ RAG pipeline ready")

        # Initialize embeddings for ingestion (separate instance with different task_type)
        app.state.ingest_embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            task_type="retrieval_document"
        )
        logger.info("✓ Ingest embeddings initialized")

        logger.info("=== Fortif.ai RAG Server Ready ===")
        logger.info(f"API Documentation: http://localhost:8000/docs")

    except Exception as e:
        logger.error(f"✗ Failed to initialize server: {e}", exc_info=True)
        raise

    yield

    # Cleanup
    logger.info("Shutting down Fortif.ai RAG Server...")
    if hasattr(app.state, "weaviate_client"):
        app.state.weaviate_client.close()
        logger.info("✓ Weaviate client closed")
    logger.info("=== Shutdown complete ===")


# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    description="RAG-powered empathetic chatbot for dementia patients",
    version=settings.api_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with structured error response."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc.detail),
            path=str(request.url.path),
            timestamp=datetime.now(timezone.utc)
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal Server Error",
            detail="An unexpected error occurred. Please try again later.",
            path=str(request.url.path),
            timestamp=datetime.now(timezone.utc)
        ).model_dump()
    )


# === API Endpoints ===

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Fortif.ai RAG API",
        "version": settings.api_version,
        "status": "operational",
        "documentation": "/docs",
        "health_check": "/api/health"
    }


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Verifies:
    - API is running
    - Weaviate connection is active
    - Collection exists and is accessible
    """
    try:
        collection = app.state.weaviate_client.collections.get(
            settings.weaviate_collection_name
        )
        collection_info = collection.aggregate.over_all(total_count=True)

        return HealthResponse(
            status="healthy",
            weaviate_connected=True,
            collection_exists=True,
            collection_count=collection_info.total_count,
            timestamp=datetime.now(timezone.utc)
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Weaviate connection failed: {str(e)}"
        )


@app.post(
    "/api/query",
    response_model=QueryResponse,
    tags=["RAG"],
    dependencies=[Depends(verify_api_key)]
)
async def query_patient(request: QueryRequest):
    """
    Query patient memories and generate empathetic response.

    This endpoint implements the full RAG pipeline:
    1. **Retrieval**: Search patient memories using vector similarity
    2. **Augmentation**: Format retrieved memories as context
    3. **Generation**: Generate empathetic response using Gemini LLM

    **Authentication**: Requires X-API-Key header

    **Example Request**:
    ```json
    {
        "patient_id": "patient_123",
        "question": "Tell me about my granddaughter's birthday",
        "include_sensitive": false,
        "emotion_filter": "positive",
        "limit": 3
    }
    ```
    """
    try:
        logger.info(f"Query received for patient_id={request.patient_id}")

        # Run RAG pipeline
        response_text, retrieved_docs = app.state.rag_pipeline.run(
            patient_id=request.patient_id,
            question=request.question,
            limit=request.limit,
            include_sensitive=request.include_sensitive,
            emotion_filter=request.emotion_filter
        )

        # Format source documents
        sources = [
            SourceDocument(
                text=doc.properties["text"],
                topic=doc.properties["topic"],
                emotion=doc.properties["emotion"],
                source=doc.properties["source"],
                distance=doc.metadata.distance,
                chunk_index=doc.properties["chunk_index"],
                total_chunks=doc.properties["total_chunks"]
            )
            for doc in retrieved_docs
        ]

        logger.info(f"Query successful: {len(sources)} sources, {len(response_text)} char response")

        return QueryResponse(
            response=response_text,
            sources=sources,
            patient_id=request.patient_id,
            metadata={
                "retrieved_count": len(retrieved_docs),
                "model": settings.llm_model,
                "temperature": settings.llm_temperature,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG pipeline failed: {str(e)}"
        )


@app.post(
    "/api/ingest",
    response_model=IngestResponse,
    tags=["Data Management"],
    dependencies=[Depends(verify_api_key)]
)
async def ingest_memory(request: IngestRequest):
    """
    Ingest new patient memory into the system.

    **Authentication**: Requires X-API-Key header (admin access)

    Performs:
    1. Validation of patient record
    2. Text chunking (500 chars, 50 overlap)
    3. Embedding generation
    4. Storage in Weaviate

    **Example Request**:
    ```json
    {
        "patient_id": "patient_123",
        "raw_text": "Jane loves her morning tea ritual...",
        "source": "family_questionnaire",
        "topic": "daily_routine",
        "is_sensitive": false,
        "entities": ["tea", "morning"],
        "emotion": "positive"
    }
    ```
    """
    try:
        logger.info(f"Ingest request received for patient_id={request.patient_id}")

        # Convert request to dict format expected by ingest logic
        patient_record = {
            "patient_id": request.patient_id,
            "raw_text": request.raw_text,
            "source": request.source,
            "topic": request.topic,
            "is_sensitive": request.is_sensitive,
            "entities": request.entities,
            "emotion": request.emotion
        }

        # Validate record
        is_valid, error_msg = validate_patient_record(patient_record)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid patient record: {error_msg}"
            )

        # Process and ingest
        stats = process_and_ingest_data(
            client=app.state.weaviate_client,
            embedding_model=app.state.ingest_embeddings,
            patient_data=[patient_record],
            batch_size=settings.default_batch_size,
            validate_records=False  # Already validated above
        )

        logger.info(f"Ingest successful: {stats}")

        return IngestResponse(
            status=stats.get("status", "success"),
            patient_id=request.patient_id,
            chunks_created=stats.get("chunks_created", 0),
            objects_upserted=stats.get("objects_upserted", 0),
            message=f"Successfully ingested memory for patient {request.patient_id}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingest failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes (development only)
        log_level="info"
    )
