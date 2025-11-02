import os
import uuid
from datetime import datetime, timezone

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Qdrant-specific imports
from qdrant_client import QdrantClient, models

# LangChain-specific imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
# --- CHANGE: Import Google's embedding model instead of OpenAI's ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- 1. Configuration ---
QDRANT_HOST = "http://localhost:6333"
QDRANT_COLLECTION_NAME = "fortif_ai_master_memory_google" # New collection name
# --- CHANGE: Define Google's model and its vector dimension ---
EMBEDDING_MODEL = "models/embedding-001"
VECTOR_DIMENSION = 768 # Critical: Google's model has 768 dimensions

def get_patient_onboarding_data():
    """Simulates receiving structured data from a family onboarding portal."""
    # This data is the same as before
    return [
        {
            "patient_id": "patient_123", "raw_text": "Jane's morning routine is very important. She wakes at 7 AM, takes her blood pressure medication, and then has tea and toast. At 8 AM, she enjoys watching the news to start her day.",
            "source": "family_questionnaire", "topic": "daily_routine", "is_sensitive": False, "entities": ["medication", "news"]
        },
        {
            "patient_id": "patient_123", "raw_text": "A cherished memory for Jane is her granddaughter Sarah's 5th birthday party. It was at the park by the old oak tree. She was so happy with the red bicycle they gave her. Her laughter was the best sound in the world.",
            "source": "family_questionnaire", "topic": "positive_memory", "is_sensitive": False, "entities": ["Sarah (granddaughter)"]
        },
        {
            "patient_id": "patient_123", "raw_text": "Jane's late husband, John, passed away in the winter of 2018. He was a wonderful man, but thinking about his final years after a long illness is still very difficult for her.",
            "source": "family_questionnaire", "topic": "family_history", "is_sensitive": True, "entities": ["John (husband)"]
        },
        {
            "patient_id": "patient_456", "raw_text": "Bill served in the army as a young man and is very proud of his service. He enjoys telling stories about his time stationed in Germany.",
            "source": "family_questionnaire", "topic": "life_history", "is_sensitive": False, "entities": ["army", "Germany"]
        },
        {
            "patient_id": "patient_456", "raw_text": "Bill needs to take his heart medication with every meal, three times a day. He sometimes forgets his lunchtime dose.",
            "source": "ehr_note", "topic": "medical_reminder", "is_sensitive": False, "entities": ["medication"]
        }
    ]

def setup_qdrant_collection(client: QdrantClient, collection_name: str):
    """Ensures the Qdrant collection is created with the correct configuration."""
    try:
        # Check if collection exists first
        if client.collection_exists(collection_name):
            print(f"Collection '{collection_name}' already exists. Deleting it first.")
            client.delete_collection(collection_name)
        
        # Create the collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                # --- CHANGE: Use the new vector dimension ---
                size=VECTOR_DIMENSION,
                distance=models.Distance.COSINE,
            ),
        )
        print(f"Collection '{collection_name}' created with vector size {VECTOR_DIMENSION}.")
    except Exception as e:
        print(f"An error occurred while creating collection: {e}")
        raise

def process_and_ingest_data(
    client: QdrantClient, 
    embedding_model: GoogleGenerativeAIEmbeddings, 
    patient_data: list,
    batch_size: int = 100
):
    """
    Processes raw data, creates embeddings, and upserts to Qdrant with batching.
    
    Args:
        client: QdrantClient instance
        embedding_model: Embedding model instance
        patient_data: List of patient records
        batch_size: Number of points to upsert per batch
    """
    if not patient_data:
        print("No data to process.")
        return
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    # Separate concerns: prepare data first
    chunks_data = _prepare_chunks(patient_data, text_splitter)
    
    if not chunks_data:
        print("No chunks generated after splitting.")
        return
    
    # Generate embeddings in batches to avoid memory issues
    print(f"Generating embeddings for {len(chunks_data)} chunks...")
    vectors = _generate_embeddings_batched(
        embedding_model, 
        [chunk['text'] for chunk in chunks_data],
        batch_size=batch_size
    )
    print("Embeddings generated successfully.")
    
    # Create Qdrant points
    qdrant_points = [
        models.PointStruct(
            id=str(uuid.uuid4()), 
            vector=vector, 
            payload=chunk
        )
        for chunk, vector in zip(chunks_data, vectors)
    ]
    
    # Batch upsert to Qdrant
    print(f"Upserting {len(qdrant_points)} points in batches of {batch_size}...")
    _batch_upsert(client, qdrant_points, batch_size)
    print("Data ingestion complete.")


def _prepare_chunks(
    patient_data: list, 
    text_splitter
) -> list:
    """Splits patient data into chunks with metadata."""
    chunks_data = []
    current_time = datetime.now(timezone.utc).isoformat()
    
    for record in patient_data:
        chunks = text_splitter.split_text(record["raw_text"])
        
        for chunk in chunks:
            chunks_data.append({
                "text": chunk,
                "patient_id": record["patient_id"],
                "source": record["source"],
                "topic": record["topic"],
                "is_sensitive": record["is_sensitive"],
                "entities": record["entities"],
                "ingested_at": current_time  # Same timestamp for all in batch
            })
    
    return chunks_data


def _generate_embeddings_batched(
    embedding_model, 
    texts: list, 
    batch_size: int = 100
) -> list:
    """Generate embeddings in batches to handle large datasets."""
    all_vectors = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"  Processing embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        vectors = embedding_model.embed_documents(batch)
        all_vectors.extend(vectors)
    
    return all_vectors


def _batch_upsert(
    client: QdrantClient, 
    points: list, 
    batch_size: int = 100
):
    """Upsert points to Qdrant in batches."""
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        print(f"  Upserting batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")
        client.upsert(
            collection_name=QDRANT_COLLECTION_NAME, 
            points=batch, 
            wait=True
        )

if __name__ == "__main__":
    print("--- Starting Fortif.ai Master Ingestion Pipeline (Google Edition) ---")
    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError("GOOGLE_API_KEY environment variable not set.")

    qdrant_client = QdrantClient(QDRANT_HOST)
    # --- CHANGE: Instantiate the Google embedding model ---
    google_embedder = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    print("Clients for Qdrant and Google AI initialized.")

    setup_qdrant_collection(qdrant_client, QDRANT_COLLECTION_NAME)
    onboarding_data = get_patient_onboarding_data()
    process_and_ingest_data(qdrant_client, google_embedder, onboarding_data)

    collection_info = qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
    print(f"\n--- Ingestion Pipeline Finished! ---")
    print(f"Total points in '{QDRANT_COLLECTION_NAME}': {collection_info.points_count}")