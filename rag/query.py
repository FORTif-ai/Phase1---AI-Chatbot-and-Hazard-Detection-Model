import os
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

import weaviate
from weaviate.classes.query import Filter, MetadataQuery
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Configuration
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
WEAVIATE_COLLECTION_NAME = "FortifAiMasterMemory"
EMBEDDING_MODEL = "models/gemini-embedding-001"  # Current model (supports 768-3072 dimensions)
VECTOR_DIMENSION = 768


def query_patient_memories(
    patient_id: str,
    question: str,
    limit: int = 3,
    include_sensitive: bool = False,
    emotion_filter: Optional[str] = None
):
    """
    Query patient memories from Weaviate.

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
    weaviate_client = weaviate.connect_to_local(
        host=WEAVIATE_HOST,
        port=WEAVIATE_PORT,
        grpc_port=WEAVIATE_GRPC_PORT
    )
    # Initialize Google embedder with task_type for queries
    embedding_model = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        task_type="retrieval_query"
    )
    print("Clients initialized.")

    try:
        print(f"\nQuerying for Patient ID: '{patient_id}'")
        print(f"Question: '{question}'")

        # Generate query vector with specified dimensionality
        print("Generating query embedding...")
        query_vector = embedding_model.embed_query(question, output_dimensionality=VECTOR_DIMENSION)

        # Build filter
        query_filter = Filter.by_property("patient_id").equal(patient_id)

        if not include_sensitive:
            query_filter = query_filter & Filter.by_property("is_sensitive").equal(False)

        if emotion_filter:
            query_filter = query_filter & Filter.by_property("emotion").equal(emotion_filter)

        # Perform search
        print("Searching Weaviate...")
        collection = weaviate_client.collections.get(WEAVIATE_COLLECTION_NAME)
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=limit,
            filters=query_filter,
            return_metadata=MetadataQuery(distance=True)
        )

        # Display results
        print("\n--- Search Results ---")
        if not response.objects:
            print("No relevant memories found.")
        else:
            print(f"Found {len(response.objects)} result(s):\n")
            for idx, obj in enumerate(response.objects, 1):
                print(f"Result {idx}:")
                print(f"  Distance: {obj.metadata.distance:.4f}")
                print(f"  Text: {obj.properties.get('text', 'N/A')}")
                print(f"  Topic: {obj.properties.get('topic', 'N/A')}")
                print(f"  Source: {obj.properties.get('source', 'N/A')}")
                print(f"  Emotion: {obj.properties.get('emotion', 'N/A')}")
                print("-" * 50)

        return response.objects

    finally:
        # Close Weaviate client connection
        weaviate_client.close()
        print("\nWeaviate client connection closed.")


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