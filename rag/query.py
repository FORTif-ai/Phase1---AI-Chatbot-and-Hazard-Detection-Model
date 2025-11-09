import os
import pprint
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

import weaviate
from weaviate_client_setup import get_weaviate_client, CLASS_NAME, VECTOR_DIMENSION

from langchain_google_genai import GoogleGenerativeAIEmbeddings

EMBEDDING_MODEL = "models/embedding-001"


def query_patient_memories(
    patient_id: str,
    question: str,
    limit: int = 3,
    include_sensitive: bool = False
) -> List[Dict[str, Any]]:
    """
    Query patient memories from Weaviate with filtering.
    
    Args:
        patient_id: The patient ID to query
        question: The natural language question
        limit: Maximum number of results to return
        include_sensitive: Whether to include sensitive memories
    
    Returns:
        List of dictionaries containing retrieved content and metadata.
    """
    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError("GOOGLE_API_KEY environment variable not set.")

    # Initialize clients
    weaviate_client = get_weaviate_client() 
    embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    print("Clients initialized.")

    print(f"\nQuerying for Patient ID: '{patient_id}'")
    print(f"Question: '{question}'")

    # Generate query vector
    print("Generating query embedding...")
    query_vector = embedding_model.embed_query(question)

    print("Building Weaviate filter...")
    
    # 1. Base filter: MUST match the specific patient_id
    patient_filter = {
        "path": ["patient_id"], 
        "operator": "Equal", 
        "valueText": patient_id
    }
    
    # 2. Safety filter: Optionally filter out sensitive memories
    sensitive_filter = None
    if not include_sensitive:
        sensitive_filter = {
            "path": ["is_sensitive"],
            "operator": "Equal",
            "valueBoolean": False
        }

    # Combine filters into a single 'where' clause
    where_filter = {
        "operator": "And",
        "operands": [patient_filter]
    }
    if sensitive_filter:
        where_filter["operands"].append(sensitive_filter)

    # 3. Perform Vector Search (GraphQL NearVector)
    print("Searching Weaviate...")
    
    # Define which properties to retrieve
    properties_to_get = [
        "content",
        "topic", 
        "source_filename", 
        "is_sensitive"
    ]
    
    search_results = (
        weaviate_client.query
        .get(CLASS_NAME, properties_to_get)
        .with_near_vector({
            "vector": query_vector
        })
        .with_where(where_filter) # Apply the personalization and safety filter
        .with_limit(limit)
        .with_additional(["distance", "id"]) # Request similarity score (distance) and object ID
        .do()
    )
    
    # 4. Process Results
    retrieved_points = []
    
    if 'data' in search_results and 'Get' in search_results['data'] and CLASS_NAME in search_results['data']['Get']:
        for item in search_results['data']['Get'][CLASS_NAME]:
            # Weaviate distance needs to be converted to similarity score (1 - distance)
            score = 1 - item['_additional']['distance']
            retrieved_points.append({
                "score": score, 
                "payload": {
                    "text": item.get('content', 'N/A'),
                    "topic": item.get('topic', 'N/A'),
                    "source": item.get('source_filename', 'N/A'),
                    "is_sensitive": item.get('is_sensitive', 'N/A'),
                    "id": item['_additional']['id']
                }
            })

    # Display results
    print("\n--- Search Results ---")
    if not retrieved_points:
        print("No relevant memories found.")
    else:
        print(f"Found {len(retrieved_points)} result(s):\n")
        for idx, result in enumerate(retrieved_points, 1):
            # Adjusted print for Weaviate result structure
            text_content = result['payload'].get('text', 'N/A')
            print(f"Result {idx}:")
            print(f"  Score: {result['score']:.4f}")
            print(f"  Text: {text_content[:100]}...")
            print(f"  Topic: {result['payload'].get('topic', 'N/A')}")
            print(f"  Source: {result['payload'].get('source', 'N/A')}")
            print(f"  Sensitive: {result['payload'].get('is_sensitive', 'N/A')}")
            print("-" * 50)
    
    # Return the compiled format for ease of integration
    return [hit['payload'] for hit in retrieved_points] 


def main():
    """Main execution function."""
    print("--- Fortif.ai Memory Retrieval System (Weaviate) ---\n")
    
    # Example query
    query_patient_memories(
        patient_id="patient_123",
        question="Tell me a happy memory about my family.",
        limit=3,
        include_sensitive=False
    )

if __name__ == "__main__":
    main()