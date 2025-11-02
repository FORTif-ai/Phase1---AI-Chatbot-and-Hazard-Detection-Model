import os
import pprint

# Qdrant-specific imports
from qdrant_client import QdrantClient, models

# LangChain-specific imports
from langchain_openai import OpenAIEmbeddings

# --- 1. Configuration ---
QDRANT_HOST = "http://localhost:6333"
QDRANT_COLLECTION_NAME = "fortif_ai_master_memory"
EMBEDDING_MODEL = "text-embedding-3-small"

def main():
    """
    Main function to run the retrieval test.
    """
    print("--- Starting Fortif.ai Retrieval Test ---")

    # 1. Check for API Key
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY environment variable not set.")

    # 2. Initialize clients
    qdrant_client = QdrantClient(QDRANT_HOST)
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    print("Clients for Qdrant and OpenAI initialized.")

    # --- 3. Define the Test Scenario ---
    # We are simulating a query from Jane Doe, whose ID is 'patient_123'.
    patient_id_to_query = "patient_123"
    
    # This is the question the user is asking.
    user_question = "Tell me a happy memory about my family."
    
    print(f"\nSimulating query for Patient ID: '{patient_id_to_query}'")
    print(f"User's Question: '{user_question}'")

    # --- 4. Vectorize the User's Question ---
    # We convert the user's question into a vector so we can find similar content.
    query_vector = embedding_model.embed_query(user_question)
    print("User question has been vectorized.")

    # --- 5. Build the Safety Filter ---
    # This is the MOST CRITICAL part of the RAG pipeline.
    # It ensures we only get data for the correct patient AND that it's not sensitive.
    query_filter = models.Filter(
        must=[
            # Condition 1: MUST belong to the correct patient.
            # This is the security wall between patient data.
            models.FieldCondition(
                key="patient_id",
                match=models.MatchValue(value=patient_id_to_query)
            ),
            
            # Condition 2: MUST NOT be marked as sensitive.
            # This is the safety net to prevent emotional triggers.
            models.FieldCondition(
                key="is_sensitive",
                match=models.MatchValue(value=False)
            )
        ]
    )
    print("Safety filter has been constructed.")

    # --- 6. Perform the Search ---
    # We now search Qdrant using the vector and the filter.
    print("Searching Qdrant for relevant and safe memories...")
    search_results = qdrant_client.search(
        collection_name=QDRANT_COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=query_filter,
        limit=3,  # Return the top 3 most relevant results
        with_payload=True  # Include the metadata in the results
    )

    # --- 7. Display the Results ---
    print("\n--- Search Results ---")
    if not search_results:
        print("No relevant and safe memories found for this query.")
    else:
        print(f"Found {len(search_results)} results:")
        # pprint is used for nicely printing the complex result object
        for result in search_results:
            pprint.pprint(result)
            print("-" * 20)

if __name__ == "__main__":
    main()