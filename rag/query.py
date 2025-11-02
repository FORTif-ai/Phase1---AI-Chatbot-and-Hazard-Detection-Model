import os
import pprint

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Qdrant-specific imports
from qdrant_client import QdrantClient, models

# --- CHANGE: Import Google's embedding model ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- 1. Configuration ---
QDRANT_HOST = "http://localhost:6333"
# --- CHANGE: Point to the new collection name ---
QDRANT_COLLECTION_NAME = "fortif_ai_master_memory_google"
EMBEDDING_MODEL = "models/embedding-001"

def main():
    print("--- Starting Fortif.ai Retrieval Test ---")

    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError("GOOGLE_API_KEY environment variable not set.")

    qdrant_client = QdrantClient(QDRANT_HOST)
    # --- CHANGE: Instantiate the Google embedding model ---
    embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    print("Clients for Qdrant and Google AI initialized.")

    # --- 3. Define the Test Scenario ---
    patient_id_to_query = "patient_123"
    user_question = "Tell me a happy memory about my family."
    
    print(f"\nSimulating query for Patient ID: '{patient_id_to_query}'")
    print(f"User's Question: '{user_question}'")

    # --- 4. Vectorize the User's Question ---
    print("Vectorizing user question with Google's model...")
    query_vector = embedding_model.embed_query(user_question)

    # --- 5. Build the Safety Filter (Logic is identical) ---
    query_filter = models.Filter(
        must=[
            models.FieldCondition(key="patient_id", match=models.MatchValue(value=patient_id_to_query)),
            models.FieldCondition(key="is_sensitive", match=models.MatchValue(value=False))
        ]
    )
    print("Safety filter constructed.")

    # --- 6. Perform the Search ---
    print("Searching Qdrant for relevant and safe memories...")
    search_results = qdrant_client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=3,
        with_payload=True
    ).points

    # --- 7. Display the Results ---
    print("\n--- Search Results ---")
    if not search_results:
        print("No relevant and safe memories found for this query.")
    else:
        print(f"Found {len(search_results)} results:")
        for result in search_results:
            pprint.pprint(result)
            print("-" * 20)

if __name__ == "__main__":
    main()