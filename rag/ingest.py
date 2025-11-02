import os
import uuid
import logging
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient, models
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
QDRANT_COLLECTION_NAME = "fortif_ai_master_memory_google"
EMBEDDING_MODEL = "models/embedding-001"
VECTOR_DIMENSION = 768
DEFAULT_BATCH_SIZE = 100
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2


class IngestionError(Exception):
    """Custom exception for ingestion pipeline errors."""
    pass


def get_patient_onboarding_data() -> List[Dict[str, Any]]:
    """Simulates receiving structured data from a family onboarding portal."""
    return [
        {
            "patient_id": "patient_123",
            "raw_text": "Jane's morning routine is very important. She wakes at 7 AM, takes her blood pressure medication, and then has tea and toast. At 8 AM, she enjoys watching the news to start her day.",
            "source": "family_questionnaire",
            "topic": "daily_routine",
            "is_sensitive": False,
            "entities": ["medication", "news"]
        },
        {
            "patient_id": "patient_123",
            "raw_text": "A cherished memory for Jane is her granddaughter Sarah's 5th birthday party. It was at the park by the old oak tree. She was so happy with the red bicycle they gave her. Her laughter was the best sound in the world.",
            "source": "family_questionnaire",
            "topic": "positive_memory",
            "is_sensitive": False,
            "entities": ["Sarah (granddaughter)"]
        },
        {
            "patient_id": "patient_123",
            "raw_text": "Jane's late husband, John, passed away in the winter of 2018. He was a wonderful man, but thinking about his final years after a long illness is still very difficult for her.",
            "source": "family_questionnaire",
            "topic": "family_history",
            "is_sensitive": True,
            "entities": ["John (husband)"]
        },
        {
            "patient_id": "patient_456",
            "raw_text": "Bill served in the army as a young man and is very proud of his service. He enjoys telling stories about his time stationed in Germany.",
            "source": "family_questionnaire",
            "topic": "life_history",
            "is_sensitive": False,
            "entities": ["army", "Germany"]
        },
        {
            "patient_id": "patient_456",
            "raw_text": "Bill needs to take his heart medication with every meal, three times a day. He sometimes forgets his lunchtime dose.",
            "source": "ehr_note",
            "topic": "medical_reminder",
            "is_sensitive": False,
            "entities": ["medication"]
        }
    ]


def validate_patient_record(record: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validates that a patient record has required fields.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = ["patient_id", "raw_text", "source", "topic", "is_sensitive", "entities"]
    
    for field in required_fields:
        if field not in record:
            return False, f"Missing required field '{field}'"
    
    if not isinstance(record["raw_text"], str) or not record["raw_text"].strip():
        return False, "Empty or invalid raw_text"
    
    return True, ""


def ensure_collection_exists(
    client: QdrantClient, 
    collection_name: str
) -> None:
    """
    Ensures the Qdrant collection exists. Creates it if it doesn't exist.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection
    """
    try:
        if client.collection_exists(collection_name):
            logger.info(f"Collection '{collection_name}' already exists. Using existing collection.")
        else:
            logger.info(f"Collection '{collection_name}' does not exist. Creating it now...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=VECTOR_DIMENSION,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info(f"Collection '{collection_name}' created successfully with vector size {VECTOR_DIMENSION}.")
            
    except Exception as e:
        logger.error(f"Failed to ensure collection exists: {e}")
        raise IngestionError(f"Collection setup failed: {e}")


def process_and_ingest_data(
    client: QdrantClient, 
    embedding_model: GoogleGenerativeAIEmbeddings, 
    patient_data: List[Dict[str, Any]],
    batch_size: int = DEFAULT_BATCH_SIZE,
    validate_records: bool = True
) -> Dict[str, Any]:
    """
    Processes raw data, creates embeddings, and upserts to Qdrant with batching.
    
    Args:
        client: QdrantClient instance
        embedding_model: Embedding model instance
        patient_data: List of patient records
        batch_size: Number of points to process per batch
        validate_records: Whether to validate records before processing
        
    Returns:
        Dictionary with ingestion statistics
    """
    if not patient_data:
        logger.warning("No data to process.")
        return {"status": "skipped", "reason": "empty_data"}
    
    # Validate records
    valid_records = []
    if validate_records:
        for record in patient_data:
            is_valid, error_msg = validate_patient_record(record)
            if is_valid:
                valid_records.append(record)
            else:
                logger.warning(
                    f"Invalid record for patient {record.get('patient_id', 'unknown')}: {error_msg}"
                )
        
        invalid_count = len(patient_data) - len(valid_records)
        if invalid_count > 0:
            logger.warning(f"Skipped {invalid_count} invalid records")
        
        patient_data = valid_records
    
    if not patient_data:
        logger.warning("No valid records to process after validation.")
        return {"status": "skipped", "reason": "no_valid_records"}
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )
    
    # Prepare chunks
    try:
        chunks_data = _prepare_chunks(patient_data, text_splitter)
    except Exception as e:
        logger.error(f"Failed to prepare chunks: {e}")
        raise IngestionError(f"Chunk preparation failed: {e}")
    
    if not chunks_data:
        logger.warning("No chunks generated after splitting.")
        return {"status": "skipped", "reason": "no_chunks_generated"}
    
    # Generate embeddings
    try:
        logger.info(f"Generating embeddings for {len(chunks_data)} chunks...")
        vectors = _generate_embeddings_batched(
            embedding_model, 
            [chunk['text'] for chunk in chunks_data],
            batch_size=batch_size
        )
        logger.info("Embeddings generated successfully.")
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        raise IngestionError(f"Embedding generation failed: {e}")
    
    # Create Qdrant points
    qdrant_points = [
        models.PointStruct(
            id=str(uuid.uuid4()), 
            vector=vector, 
            payload=chunk
        )
        for chunk, vector in zip(chunks_data, vectors)
    ]
    
    # Batch upsert
    try:
        logger.info(f"Upserting {len(qdrant_points)} points in batches of {batch_size}...")
        _batch_upsert(client, qdrant_points, batch_size)
        logger.info("Data ingestion complete.")
    except Exception as e:
        logger.error(f"Failed to upsert points: {e}")
        raise IngestionError(f"Upsert failed: {e}")
    
    return {
        "status": "success",
        "records_processed": len(patient_data),
        "chunks_created": len(chunks_data),
        "points_upserted": len(qdrant_points)
    }


def _prepare_chunks(
    patient_data: List[Dict[str, Any]], 
    text_splitter: RecursiveCharacterTextSplitter
) -> List[Dict[str, Any]]:
    """Splits patient data into chunks with metadata."""
    chunks_data = []
    current_time = datetime.now(timezone.utc).isoformat()
    
    for record in patient_data:
        try:
            chunks = text_splitter.split_text(record["raw_text"])
            
            for idx, chunk in enumerate(chunks):
                chunks_data.append({
                    "text": chunk,
                    "patient_id": record["patient_id"],
                    "source": record["source"],
                    "topic": record["topic"],
                    "is_sensitive": record["is_sensitive"],
                    "entities": record["entities"],
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "ingested_at": current_time
                })
        except Exception as e:
            logger.error(f"Failed to chunk record for patient {record.get('patient_id', 'unknown')}: {e}")
            continue
    
    return chunks_data


def _generate_embeddings_batched(
    embedding_model: GoogleGenerativeAIEmbeddings, 
    texts: List[str], 
    batch_size: int = DEFAULT_BATCH_SIZE
) -> List[List[float]]:
    """Generate embeddings in batches to handle large datasets."""
    all_vectors = []
    total_batches = (len(texts) - 1) // batch_size + 1
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        try:
            logger.info(f"  Processing embedding batch {batch_num}/{total_batches}")
            vectors = embedding_model.embed_documents(batch)
            
            if vectors and len(vectors[0]) != VECTOR_DIMENSION:
                raise IngestionError(
                    f"Vector dimension mismatch: expected {VECTOR_DIMENSION}, got {len(vectors[0])}"
                )
            
            all_vectors.extend(vectors)
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings for batch {batch_num}: {e}")
            raise
    
    return all_vectors


def _batch_upsert(
    client: QdrantClient, 
    points: List[models.PointStruct], 
    batch_size: int = DEFAULT_BATCH_SIZE
) -> None:
    """Upsert points to Qdrant in batches with retry logic."""
    total_batches = (len(points) - 1) // batch_size + 1
    
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(f"  Upserting batch {batch_num}/{total_batches}")
                client.upsert(
                    collection_name=QDRANT_COLLECTION_NAME, 
                    points=batch, 
                    wait=True
                )
                break
                
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    logger.warning(
                        f"Batch {batch_num} failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}. "
                        f"Retrying in {RETRY_DELAY}s..."
                    )
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"Batch {batch_num} failed after {MAX_RETRIES} attempts: {e}")
                    raise


def initialize_clients() -> Tuple[QdrantClient, GoogleGenerativeAIEmbeddings]:
    """Initialize and return Qdrant and embedding clients."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY environment variable not set.")
    
    try:
        qdrant_client = QdrantClient(QDRANT_HOST)
        google_embedder = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        logger.info("Clients for Qdrant and Google AI initialized.")
        return qdrant_client, google_embedder
    except Exception as e:
        logger.error(f"Failed to initialize clients: {e}")
        raise


def main() -> None:
    """Main execution function."""
    logger.info("--- Starting Fortif.ai Data Ingestion ---")
    
    try:
        qdrant_client, google_embedder = initialize_clients()
        ensure_collection_exists(qdrant_client, QDRANT_COLLECTION_NAME)
        
        onboarding_data = get_patient_onboarding_data()
        stats = process_and_ingest_data(qdrant_client, google_embedder, onboarding_data)
        
        collection_info = qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        
        logger.info("--- Ingestion Complete! ---")
        logger.info(f"Ingestion stats: {stats}")
        logger.info(f"Total points in '{QDRANT_COLLECTION_NAME}': {collection_info.points_count}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()