import os
import uuid
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional
from weaviate.collections.classes.data import DataObject

from dotenv import load_dotenv
load_dotenv()

import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType, Configure
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
WEAVIATE_COLLECTION_NAME = "FortifAiMasterMemory"
EMBEDDING_MODEL = "models/gemini-embedding-001"  # Current model (supports 768-3072 dimensions)
VECTOR_DIMENSION = 768
DEFAULT_BATCH_SIZE = 100
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


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
            "entities": ["medication", "news"],
            "emotion": "neutral"
        },
        {
            "patient_id": "patient_123",
            "raw_text": "A cherished memory for Jane is her granddaughter Sarah's 5th birthday party. It was at the park by the old oak tree. She was so happy with the red bicycle they gave her. Her laughter was the best sound in the world.",
            "source": "family_questionnaire",
            "topic": "positive_memory",
            "is_sensitive": False,
            "entities": ["Sarah (granddaughter)"],
            "emotion": "positive"
        },
        {
            "patient_id": "patient_123",
            "raw_text": "Jane's late husband, John, passed away in the winter of 2018. He was a wonderful man, but thinking about his final years after a long illness is still very difficult for her.",
            "source": "family_questionnaire",
            "topic": "family_history",
            "is_sensitive": True,
            "entities": ["John (husband)"],
            "emotion": "negative"
        },
        {
            "patient_id": "patient_456",
            "raw_text": "Bill served in the army as a young man and is very proud of his service. He enjoys telling stories about his time stationed in Germany.",
            "source": "family_questionnaire",
            "topic": "life_history",
            "is_sensitive": False,
            "entities": ["army", "Germany"],
            "emotion": "positive"
        },
        {
            "patient_id": "patient_456",
            "raw_text": "Bill needs to take his heart medication with every meal, three times a day. He sometimes forgets his lunchtime dose.",
            "source": "ehr_note",
            "topic": "medical_reminder",
            "is_sensitive": False,
            "entities": ["medication"],
            "emotion": "neutral"
        }
    ]


def validate_patient_record(record: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validates that a patient record has required fields.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = ["patient_id", "raw_text", "source", "topic", "is_sensitive", "entities", "emotion"]
    
    for field in required_fields:
        if field not in record:
            return False, f"Missing required field '{field}'"
    
    if not isinstance(record["raw_text"], str) or not record["raw_text"].strip():
        return False, "Empty or invalid raw_text"
    
    # Validate emotion field
    valid_emotions = ["positive", "negative", "neutral", "mixed", "unknown"]
    if record.get("emotion") not in valid_emotions:
        return False, f"Invalid emotion value. Must be one of: {', '.join(valid_emotions)}"
    
    return True, ""


def ensure_collection_exists(
    client: weaviate.WeaviateClient,
    collection_name: str
) -> None:
    """
    Ensures the Weaviate collection exists. Creates it if it doesn't exist.

    Args:
        client: WeaviateClient instance
        collection_name: Name of the collection
    """
    try:
        if client.collections.exists(collection_name):
            logger.info(f"Collection '{collection_name}' already exists. Using existing collection.")
        else:
            logger.info(f"Collection '{collection_name}' does not exist. Creating it now...")
            client.collections.create(
                name=collection_name,
                vector_config=Configure.Vectors.self_provided(),
                properties=[
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="patient_id", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="topic", data_type=DataType.TEXT),
                    Property(name="is_sensitive", data_type=DataType.BOOL),
                    Property(name="entities", data_type=DataType.TEXT_ARRAY),
                    Property(name="emotion", data_type=DataType.TEXT),
                    Property(name="chunk_index", data_type=DataType.INT),
                    Property(name="total_chunks", data_type=DataType.INT),
                    Property(name="ingested_at", data_type=DataType.TEXT),
                ],
            )
            logger.info(f"Collection '{collection_name}' created successfully with vector size {VECTOR_DIMENSION}.")

    except Exception as e:
        logger.error(f"Failed to ensure collection exists: {e}")
        raise IngestionError(f"Collection setup failed: {e}")


def process_and_ingest_data(
    client: weaviate.WeaviateClient,
    embedding_model: GoogleGenerativeAIEmbeddings,
    patient_data: List[Dict[str, Any]],
    batch_size: int = DEFAULT_BATCH_SIZE,
    validate_records: bool = True
) -> Dict[str, Any]:
    """
    Processes raw data, creates embeddings, and upserts to Weaviate with batching.

    Args:
        client: WeaviateClient instance
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
    
    # Create Weaviate data objects
    data_objects = [
        wvc.data.DataObject(
            properties={
                "text": chunk["text"],
                "patient_id": chunk["patient_id"],
                "source": chunk["source"],
                "topic": chunk["topic"],
                "is_sensitive": chunk["is_sensitive"],
                "entities": chunk["entities"],
                "emotion": chunk["emotion"],
                "chunk_index": chunk["chunk_index"],
                "total_chunks": chunk["total_chunks"],
                "ingested_at": chunk["ingested_at"],
            },
            vector=vector
        )
        for chunk, vector in zip(chunks_data, vectors)
    ]

    # Batch upsert using Weaviate's built-in batch manager
    try:
        collection = client.collections.get(WEAVIATE_COLLECTION_NAME)
        logger.info(f"Upserting {len(data_objects)} objects...")

        # Use dynamic batching - the client handles batching, errors, and retries automatically
        with collection.batch.dynamic() as batch:
            for idx, data_obj in enumerate(data_objects, 1):
                batch.add_object(
                    properties=data_obj.properties,
                    vector=data_obj.vector
                )

                # Log progress every 100 objects
                if idx % 100 == 0:
                    logger.info(f"  Processed {idx}/{len(data_objects)} objects")

        # Check for any failed objects after batch completion
        failed_objs = collection.batch.failed_objects
        if failed_objs:
            logger.warning(
                f"Ingestion completed with {len(failed_objs)} failed objects out of {len(data_objects)}"
            )
            # Log details of first few failures for debugging
            for failed in failed_objs[:3]:
                logger.warning(f"  Failed object: {failed}")
        else:
            logger.info(f"Successfully imported all {len(data_objects)} objects")

        logger.info("Data ingestion complete.")
    except Exception as e:
        logger.error(f"Failed to upsert objects: {e}")
        raise IngestionError(f"Upsert failed: {e}")

    return {
        "status": "success",
        "records_processed": len(patient_data),
        "chunks_created": len(chunks_data),
        "objects_upserted": len(data_objects)
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
                    "emotion": record["emotion"],
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
            # Pass output_dimensionality as a method parameter, not constructor parameter
            vectors = embedding_model.embed_documents(batch, output_dimensionality=VECTOR_DIMENSION)

            if vectors and len(vectors[0]) != VECTOR_DIMENSION:
                raise IngestionError(
                    f"Vector dimension mismatch: expected {VECTOR_DIMENSION}, got {len(vectors[0])}"
                )

            all_vectors.extend(vectors)

        except Exception as e:
            logger.error(f"Failed to generate embeddings for batch {batch_num}: {e}")
            raise

    return all_vectors


def initialize_clients() -> Tuple[weaviate.WeaviateClient, GoogleGenerativeAIEmbeddings]:
    """Initialize and return Weaviate and embedding clients."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY environment variable not set.")

    try:
        # Connect to local Weaviate instance
        weaviate_client = weaviate.connect_to_local(
            host=WEAVIATE_HOST,
            port=WEAVIATE_PORT,
            grpc_port=WEAVIATE_GRPC_PORT
        )
        # Initialize Google embedder with task_type
        google_embedder = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            task_type="retrieval_document"
        )
        logger.info("Clients for Weaviate and Google AI initialized.")
        return weaviate_client, google_embedder
    except Exception as e:
        logger.error(f"Failed to initialize clients: {e}")
        raise


def main() -> None:
    """Main execution function."""
    logger.info("--- Starting Fortif.ai Data Ingestion ---")

    weaviate_client = None
    try:
        weaviate_client, google_embedder = initialize_clients()
        ensure_collection_exists(weaviate_client, WEAVIATE_COLLECTION_NAME)

        onboarding_data = get_patient_onboarding_data()
        stats = process_and_ingest_data(weaviate_client, google_embedder, onboarding_data)

        # Get collection information
        collection = weaviate_client.collections.get(WEAVIATE_COLLECTION_NAME)
        collection_info = collection.aggregate.over_all(total_count=True)

        logger.info("--- Ingestion Complete! ---")
        logger.info(f"Ingestion stats: {stats}")
        logger.info(f"Total objects in '{WEAVIATE_COLLECTION_NAME}': {collection_info.total_count}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise
    finally:
        # Close Weaviate client connection
        if weaviate_client is not None:
            weaviate_client.close()
            logger.info("Weaviate client connection closed.")


if __name__ == "__main__":
    main()