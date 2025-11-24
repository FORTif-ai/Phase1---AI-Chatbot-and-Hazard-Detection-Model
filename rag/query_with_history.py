import os
from typing import List
from dotenv import load_dotenv
load_dotenv()

import weaviate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from rag_pipeline import RAGPipeline
from models import Message

# Configuration
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash"

print("Script started. Importing modules...")

def main():
    print("--- Verifying Session History & Query Rewriting ---\n")

    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError("GOOGLE_API_KEY environment variable not set.")

    # Initialize clients
    weaviate_client = weaviate.connect_to_local(
        host=WEAVIATE_HOST,
        port=WEAVIATE_PORT,
        grpc_port=WEAVIATE_GRPC_PORT
    )
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        task_type="retrieval_query"
    )
    
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0
    )

    pipeline = RAGPipeline(weaviate_client, embeddings, llm)
    print("Pipeline initialized.")

    try:
        patient_id = "patient_123"
        
        # Turn 1: Initial Question
        q1 = "Who is Sarah?"
        print(f"\n--- Turn 1: '{q1}' ---")
        resp1, docs1 = pipeline.run(patient_id=patient_id, question=q1)
        print(f"Response: {resp1}")
        
        # Turn 2: Follow-up Question (Ambiguous)
        q2 = "How old is she?"
        history = [
            Message(role="user", content=q1),
            Message(role="assistant", content=resp1)
        ]
        print(f"\n--- Turn 2: '{q2}' (with history) ---")
        print(f"History: {[msg.content for msg in history]}")
        
        # This should trigger query rewriting: "How old is she?" -> "How old is Sarah?"
        resp2, docs2 = pipeline.run(patient_id=patient_id, question=q2, history=history)
        
        print(f"Response: {resp2}")
        
        # Verify that we retrieved documents about Sarah
        print("\n--- Verification ---")
        sarah_docs = [d for d in docs2 if "Sarah" in d.properties["text"]]
        if sarah_docs:
            print("SUCCESS: Retrieved documents containing 'Sarah' for the query 'How old is she?'")
        else:
            print("FAILURE: Did not retrieve documents about Sarah.")

    finally:
        weaviate_client.close()
        print("\nWeaviate client connection closed.")

if __name__ == "__main__":
    main()
