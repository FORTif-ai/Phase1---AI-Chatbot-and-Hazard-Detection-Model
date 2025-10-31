import os
import uuid
from datetime import datetime, timezone

# Qdrant-specific imports
from qdrant_client import QdrantClient, models

# LangChain-specific imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# --- 1. Configuration ---
# Centralized configuration for easy management.
QDRANT_HOST = "http://localhost:6333"
QDRANT_COLLECTION_NAME = "fortif_ai_master_memory"
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_DIMENSION = 1536 # Dimension for the 'text-embedding-3-small' model

def get_patient_onboarding_data():
    """
    Simulates receiving structured data from a family onboarding portal.
    In a real application, this data would come from a secure API endpoint.
    """
    return [
        {
            "patient_id": "patient_123", # Jane Doe
            "raw_text": "Jane's morning routine is very important. She wakes at 7 AM, takes her blood pressure medication, and then has tea and toast. At 8 AM, she enjoys watching the news to start her day.",
            "source": "family_questionnaire",
            "topic": "daily_routine",
            "is_sensitive": False,
            "entities": ["medication", "news"]
        },
        {
            "patient_id": "patient_123", # Jane Doe
            "raw_text": "A cherished memory for Jane is her granddaughter Sarah's 5th birthday party. It was at the park by the old oak tree. She was so happy with the red bicycle they gave her. Her laughter was the best sound in the world.",
            "source": "family_questionnaire",
            "topic": "positive_memory",
            "is_sensitive": False,
            "entities": ["Sarah (granddaughter)"]
        },
        {
            "patient_id": "patient_123", # Jane Doe
            "raw_text": "Jane's late husband, John, passed away in the winter of 2018. He was a wonderful man, but thinking about his final years after a long illness is still very difficult for her.",
            "source": "family_questionnaire",
            "topic": "family_history",
            "is_sensitive": True,
            "entities": ["John (husband)"]
        },
        {
            "patient_id": "patient_456", # Bill Smith
            "raw_text": "Bill served in the army as a young man and is very proud of his service. He enjoys telling stories about his time stationed in Germany.",
            "source": "family_questionnaire",
            "topic": "life_history",
            "is_sensitive": False,
            "entities": ["army", "Germany"]
        },
        {
            "patient_id": "patient_456", # Bill Smith
            "raw_text": "Bill needs to take his heart medication with every meal, three times a day. He sometimes forgets his lunchtime dose.",
            "source": "ehr_note",
            "topic": "medical_reminder",
            "is_sensitive": False,
            "entities": ["medication"]
        }
    ]

def setup_qdrant_collection(client: QdrantClient, collection_name: str):
    """Ensures the Qdrant collection is created with the correct configuration."""
    try:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=VECTOR_DIMENSION,
                distance=models.Distance.COSINE,
            ),
        )
        print(f"Collection '{collection_name}' created successfully.")
    except Exception as e:
        print(f"An error occurred while creating collection: {e}")
        raise

def process_and_ingest_data(client: QdrantClient, embedding_model: OpenAIEmbeddings, patient_data: list):
    """
    Processes the raw data, creates embeddings, and upserts it into Qdrant.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    points_to_create = []
    texts_to_embed = []

    print(f"\nProcessing {len(patient_data)} raw data records...")
    for record in patient_data:
        # Split the raw text into smaller, more manageable chunks
        chunks = text_splitter.split_text(record["raw_text"])
        
        for chunk in chunks:
            # Prepare the metadata payload for this chunk
            payload = {
                "text": chunk,
                "patient_id": record["patient_id"],
                "source": record["source"],
                "topic": record["topic"],
                "is_sensitive": record["is_sensitive"],
                "entities": record["entities"],
                "ingested_at": datetime.now(timezone.utc).isoformat()
            }
            points_to_create.append(payload)
            texts_to_embed.append(chunk)

    print(f"Created {len(points_to_create)} total chunks to be vectorized.")

    # Generate embeddings for all chunks in a single batch call for efficiency
    print("Generating vector embeddings for all chunks...")
    vectors = embedding_model.embed_documents(texts_to_embed)
    print("Embeddings generated successfully.")

    # Prepare the final PointStructs for Qdrant
    qdrant_points = [
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload=point,
        )
        for point, vector in zip(points_to_create, vectors)
    ]

    # Upsert all the points to Qdrant in a single batch
    print(f"Upserting {len(qdrant_points)} points into Qdrant...")
    client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=qdrant_points,
        wait=True,
    )
    print("Data ingestion complete.")


# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Starting Fortif.ai Master Ingestion Pipeline ---")

    # 1. Check for API Key
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY environment variable not set. Please export it before running.")

    # 2. Initialize clients
    qdrant_client = QdrantClient(QDRANT_HOST)
    openai_embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    print("Clients for Qdrant and OpenAI initialized.")

    # 3. Set up the Qdrant collection
    setup_qdrant_collection(qdrant_client, QDRANT_COLLECTION_NAME)

    # 4. Get the simulated patient data
    onboarding_data = get_patient_onboarding_data()
    print(f"Fetched {len(onboarding_data)} records from simulated onboarding source.")

    # 5. Process the data and ingest it into Qdrant
    process_and_ingest_data(qdrant_client, openai_embedder, onboarding_data)

    # 6. Final verification
    collection_info = qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
    point_count = collection_info.points_count
    
    print("\n--- Ingestion Pipeline Finished! ---")
    print(f"Total points in '{QDRANT_COLLECTION_NAME}': {point_count}")
    print("You can now query this data using the patient_id and is_sensitive filters.")