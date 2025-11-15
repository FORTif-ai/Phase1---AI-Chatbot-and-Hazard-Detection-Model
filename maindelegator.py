from typing import Optional, List, Dict, Any
from convo import LLMService
import weaviate
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import os
from dotenv import load_dotenv
load_dotenv()

from rag.weaviate_client_setup import get_weaviate_client, CLASS_NAME

# --- WEAVIATE CONFIGURATION ---
EMBEDDING_MODEL = "models/embedding-001"
CONF_THRESHOLD = 0.7  # Define a confidence threshold for routing

def execute_weaviate_search(
    client: weaviate.Client, 
    query_vector: List[float], 
    limit: int, 
    where_filter: Optional[Dict] = None
) -> List[Dict[str, Any]]:
    """Helper function to execute the Weaviate search and process results."""
    
    # Define which properties to retrieve
    properties_to_get = [
        "content", 
        "patient_id", 
        "is_sensitive", 
        "source_filename", 
        "topic"
    ]
    
    query_builder = (
        client.query
        .get(CLASS_NAME, properties_to_get)
        .with_near_vector({
            "vector": query_vector
        })
        .with_limit(limit)
        .with_additional(["distance", "id"])
    )
    
    if where_filter:
        query_builder = query_builder.with_where(where_filter)
        
    results = query_builder.do()

    # Process results into a list of simplified hit objects
    processed_results = []
    if 'data' in results and 'Get' in results['data'] and CLASS_NAME in results['data']['Get']:
        for item in results['data']['Get'][CLASS_NAME]:
            # Weaviate distance needs to be converted to similarity score (1 - distance)
            # 0 = identical vectors, 2 = polar opposites. Similarity = 1 - (distance / 2) if distance uses dot product.
            # Assuming COSINE distance (which Weaviate uses by default or as configured), 
            # similarity is calculated as 1 - distance/2 (if distance is normalized). 
            # Since the data uses COSINE (which is usually 0 to 2), we use (1 - distance/2)
            # However, for simplicity and alignment with Qdrant's cosine scoring (0 to 1, 1 is best), 
            # we'll stick to 1 - distance, assuming the vectors are normalized for a standard cosine distance where 0 is perfect.
            # *Correction*: Weaviate's GraphQL result for cosine distance typically returns 0 (perfect match) to 2 (worst match).
            # The simplified score is 1 - distance/2, but for routing thresholding, just checking the raw distance is often fine.
            # Sticking to Qdrant's concept where high score is better, let's use:
            score = 1 - item['_additional']['distance']
            
            # Since cosine similarity ranges from -1 (worst) to 1 (best), 
            # and Weaviate converts this to a distance (0 to 2), 
            # a perfect score of 1.0 would map to distance=0. 
            # A score of 0.0 would map to distance=1. A score of -1.0 would map to distance=2.
            # Let's trust the distance is the best metric for routing similarity.
            
            processed_results.append({
                "patient_id": item.get('patient_id'),
                "score": score, # Using 1 - distance for a 1.0 (best) to -1.0 (worst) similarity feel
                "text": item.get('content'),
                "is_sensitive": item.get('is_sensitive'),
                "source": item.get('source_filename'),
            })
            
    return processed_results

def get_personalized_context(
    query_command: str, 
    patient_id: str,
    embedder: GoogleGenerativeAIEmbeddings,
    client: weaviate.Client,
    top_k: int = 5
) -> str:
    """
    Retrieves the most relevant information for a specific patient from Weaviate.
    """
    try:
        print(f"Retrieving {top_k} personalized information for Patient {patient_id}...")
        
        # 1. Convert the user command into a query vector
        query_vector = embedder.embed_query(query_command)

        # 2. Create the personalization filter (must match the patient_id)
        personal_filter = {
            "operator": "And",
            "operands": [
                # Filter 1: MUST match the patient_id
                {
                    "path": ["patient_id"],
                    "operator": "Equal",
                    "valueText": patient_id
                },
                # Filter 2: MUST NOT be sensitive
            ]
        }

        # 3. Search Weaviate with the filter
        search_results = execute_weaviate_search(
            client=client,
            query_vector=query_vector, 
            limit=top_k,
            where_filter=personal_filter
        )
        
        # 4. Extract and Compile Context
        print(f"search_results: {search_results}")
        context_chunks = [hit.get('text', '') for hit in search_results]
            
        if not context_chunks:
            return f"No specific memories found for patient {patient_id} regarding the query."

        # Compile context string
        context_string = "\n---\n".join(context_chunks)
        return "\n\n--- PERSONALIZED MEMORY CONTEXT ---\n" + context_string

    except Exception as e:
        print(f"Error during Weaviate context retrieval: {e}")

class delegation:
    @staticmethod
    async def delegate(command: str, patient_id: str) -> Dict[str, Any]:
        weaviate_client = get_weaviate_client()
        google_embedder = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        
        # 1. RETRIEVE PERSONALIZED CONTEXT (Approach 1)
        personalized_context = get_personalized_context(
            query_command=command,
            patient_id=patient_id, # Pass patient_id
            embedder=google_embedder,
            client=weaviate_client, # Passed new client
            top_k=5
        )

        print(f"--- Personalized Context Retrieved ---{personalized_context}")
                    
        # 2. CONSTRUCT THE FINAL LLM PROMPT
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
        
        # 3. INTERACT WITH THE LLM
        llm_service = LLMService(full_system_prompt=FINAL_LLM_CONTEXT)

        result = llm_service.stream_reply(command)

        response = "Response: "
        for chunk in result:
            response += chunk

        print(response)