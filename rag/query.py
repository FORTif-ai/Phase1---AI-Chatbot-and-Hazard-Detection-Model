import os

from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient, models
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
QDRANT_COLLECTION_NAME = "fortif_ai_master_memory_google"
EMBEDDING_MODEL = "models/embedding-001"


def query_patient_memories(
    patient_id: str,
    question: str,
    limit: int = 3,
    include_sensitive: bool = False,
    emotion_filter: str = None
):
    """
    Query patient memories from Qdrant.
    
    Args:
        patient_id: The patient ID to query
        question: The natural language question
        limit: Maximum number of results to return
        include_sensitive: Whether to include sensitive memories
        emotion_filter: Optional emotion filter ('positive', 'negative', 'neutral', 'mixed')
    """
    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError("GOOGLE_API_KEY environment variable not set.")

    # Initialize clients
    qdrant_client = QdrantClient(QDRANT_HOST)
    embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    print("Clients initialized.")

    print(f"\nQuerying for Patient ID: '{patient_id}'")
    print(f"Question: '{question}'")

    # Generate query vector
    print("Generating query embedding...")
    query_vector = embedding_model.embed_query(question)

    # Build safety filter
    filter_conditions = [
        models.FieldCondition(key="patient_id", match=models.MatchValue(value=patient_id))
    ]
    
    if not include_sensitive:
        filter_conditions.append(
            models.FieldCondition(key="is_sensitive", match=models.MatchValue(value=False))
        )
    
    if emotion_filter:
        filter_conditions.append(
            models.FieldCondition(key="emotion", match=models.MatchValue(value=emotion_filter))
        )
    
    query_filter = models.Filter(must=filter_conditions)

    # Perform search
    print("Searching Qdrant...")
    search_results = qdrant_client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True
    ).points

    # Display results
    print("\n--- Search Results ---")
    if not search_results:
        print("No relevant memories found.")
    else:
        print(f"Found {len(search_results)} result(s):\n")
        for idx, result in enumerate(search_results, 1):
            print(f"Result {idx}:")
            print(f"  Score: {result.score:.4f}")
            print(f"  Text: {result.payload.get('text', 'N/A')}")
            print(f"  Topic: {result.payload.get('topic', 'N/A')}")
            print(f"  Source: {result.payload.get('source', 'N/A')}")
            print(f"  Emotion: {result.payload.get('emotion', 'N/A')}")
            print("-" * 50)
    
    return search_results


def main():
    """Main execution function."""
    print("--- Fortif.ai Memory Retrieval System ---\n")
    
    # Example query
    query_patient_memories(
        patient_id="patient_123",
        question="Tell me a happy memory about my family.",
        limit=3,
        include_sensitive=False
    )

if __name__ == "__main__":
    main()