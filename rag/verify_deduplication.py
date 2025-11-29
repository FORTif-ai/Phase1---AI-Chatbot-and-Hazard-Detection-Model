import os
import logging
from dotenv import load_dotenv
load_dotenv()

import weaviate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from ingest import process_and_ingest_data, initialize_clients, WEAVIATE_COLLECTION_NAME

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("--- Verifying Duplicate Detection ---\n")

    weaviate_client = None
    try:
        weaviate_client, google_embedder = initialize_clients()
        collection = weaviate_client.collections.get(WEAVIATE_COLLECTION_NAME)
        
        # 1. Get initial count
        initial_count = collection.aggregate.over_all(total_count=True).total_count
        print(f"Initial object count: {initial_count}")

        # 2. Define a test memory
        test_memory = [{
            "patient_id": "test_patient_dedup",
            "raw_text": "This is a unique test memory to verify deduplication logic. It should only appear once.",
            "source": "test_script",
            "topic": "testing",
            "is_sensitive": False,
            "entities": ["test"],
            "emotion": "neutral"
        }]

        # 3. First Ingestion
        print("\n--- Run 1: Ingesting test memory ---")
        process_and_ingest_data(weaviate_client, google_embedder, test_memory)
        
        count_after_run1 = collection.aggregate.over_all(total_count=True).total_count
        print(f"Count after Run 1: {count_after_run1}")
        
        expected_increase = 1 # Assuming it splits into 1 chunk
        if count_after_run1 != initial_count + expected_increase:
            print(f"WARNING: Count increased by {count_after_run1 - initial_count}, expected {expected_increase}")

        # 4. Second Ingestion (Duplicate)
        print("\n--- Run 2: Ingesting SAME test memory again ---")
        process_and_ingest_data(weaviate_client, google_embedder, test_memory)
        
        count_after_run2 = collection.aggregate.over_all(total_count=True).total_count
        print(f"Count after Run 2: {count_after_run2}")

        # 5. Verification
        if count_after_run2 == count_after_run1:
            print("\nSUCCESS: Object count did not increase. Deduplication is working!")
        else:
            print(f"\nFAILURE: Object count increased by {count_after_run2 - count_after_run1}. Deduplication failed.")

        # Cleanup (optional, but good practice for tests)
        # collection.data.delete_many(
        #     where=weaviate.classes.query.Filter.by_property("patient_id").equal("test_patient_dedup")
        # )

    except Exception as e:
        logger.error(f"Verification failed: {e}")
    finally:
        if weaviate_client:
            weaviate_client.close()

if __name__ == "__main__":
    main()
