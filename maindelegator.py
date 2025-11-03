from typing import Optional
from convo import LLMService
from qdrant_client import QdrantClient, models
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import os
from dotenv import load_dotenv
load_dotenv()

# --- QDRANT CONFIGURATION ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
QDRANT_COLLECTION_NAME = "fortif_ai_master_memory_google"
EMBEDDING_MODEL = "models/embedding-001"
CONF_THRESHOLD = 0.7  # Define a confidence threshold for routing

def route_to_patient_id(
    query_command: str, 
    embedder: GoogleGenerativeAIEmbeddings,
    client: QdrantClient
) -> Optional[str]:
    """
    Uses vector search on the master memory collection to find the most likely 
    patient_id associated with the command, replacing the hardcoded logic.
    """
    try:
        print("Routing command to a patient ID via Qdrant search...")
        
        # 1. Convert the command into a query vector
        query_vector = embedder.embed_query(query_command)

        # 2. Search the Master Memory Collection for the best match
        # We search with no filters, focusing only on semantic similarity.
        search_results = client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=query_vector,
            limit=1,  # Only need the single best match to get the ID
            with_payload=True 
        )

        if not search_results:
            print("No matching results found in Qdrant for routing.")
            return None

        # 3. Extract the best patient ID and score
        best_hit = search_results[0]
        print(f"This is the most likely result: {best_hit}")
        best_patient_id = best_hit.payload.get('patient_id')
        best_score = best_hit.score
        
        # 4. Apply the confidence threshold logic
        if best_score < CONF_THRESHOLD:
            print(f"Low confidence ({best_score:.2f}) — defaulting to a general persona.")
            return "" 
        else:
            print(f"Qdrant routed command to Patient ID: {best_patient_id} (Score: {best_score:.2f})")
            return best_patient_id

    except Exception as e:
        print(f"Error during Qdrant routing: {e}")
        return None

def get_personalized_context(
    query_command: str, 
    patient_id: str,
    embedder: GoogleGenerativeAIEmbeddings,
    client: QdrantClient,
    top_k: int = 5
) -> str:
    """
    Retrieves the most relevant information for a specific patient from Qdrant.
    """
    try:
        print(f"Retrieving {top_k} personalized information for Patient {patient_id}...")
        
        # 1. Convert the user command into a query vector
        query_vector = embedder.embed_query(query_command)

        # 2. Create the personalization filter (must match the patient_id)
        personal_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="patient_id",
                    match=models.MatchValue(value=patient_id)
                ),
                models.FieldCondition(
                    key="is_sensitive",
                    match=models.MatchValue(value=False) # exclude sensitive points
                )
            ]
        )

        # 3. Search Qdrant with the filter
        search_results = client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=personal_filter, 
            limit=top_k,
            with_payload=True
        )
        
        # 4. Extract and Compile Context
        print(f"search_results: {search_results}")
        context_chunks = [hit.payload.get('text', '') for hit in search_results]
            
        if not context_chunks:
            return f"No specific memories found for patient {patient_id} regarding the query."

        # Compile context string
        return "\n\n--- PERSONALIZED MEMORY CONTEXT ---\n" + "\n---\n".join(context_chunks)

    except Exception as e:
        print(f"Error during Qdrant context retrieval: {e}")
        return "System error: Could not retrieve personalized context."

class delegation:
    @staticmethod
    async def delegate(command):
        qdrant_client = QdrantClient(QDRANT_HOST)
        google_embedder = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        
        # 1. DYNAMIC ROUTING
        TARGET_PATIENT_ID = route_to_patient_id(
            query_command=command,
            embedder=google_embedder,
            client=qdrant_client
        )
        
        if not TARGET_PATIENT_ID:
            print("❌ Delegation failed. Using a generic response.")
            # Fallback to a generic response or an error message
            FINAL_LLM_CONTEXT = "No patient identity could be determined. Please ask a general question."
        else:
            # 2. RETRIEVE PERSONALIZED CONTEXT (Approach 1)
            personalized_context = get_personalized_context(
                query_command=command,
                patient_id=TARGET_PATIENT_ID,
                embedder=google_embedder,
                client=qdrant_client,
                top_k=5
            )

            print(f"--- Personalized Context Retrieved ---{personalized_context}")
                        
            # 3. CONSTRUCT THE FINAL LLM PROMPT
            SYSTEM_PROMPT = f"""
            You are Fortif.ai, a kind, patient, and caring AI companion for seniors. Your role is to be helpful and supportive.

            Strict Rules:
            You MUST ONLY use the information provided in the "Context" section below to answer the user's question. Do not make up information.
            If the context does not contain the answer, simply say "I'm not sure I have that information, but I'm here to help with anything else."
            NEVER mention sensitive topics. Keep the conversation positive and safe.
            Keep your answers concise and easy to understand.

            ---
            Context:
            {personalized_context} <--- this is the ("R" in 'RAG')
            """
            
            FINAL_LLM_CONTEXT = SYSTEM_PROMPT

        print("--- LLM Input Context Generated ---")
        
        # 4. INTERACT WITH THE LLM
        llm_service = LLMService(full_system_prompt=FINAL_LLM_CONTEXT)

        result = llm_service.stream_reply(command)

        response = "Response: "
        for chunk in result:
            response += chunk

        print(response)