from typing import Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import base64
from langchain_core.messages import HumanMessage
import whisper
import httpx

# Import services
from mycalendar import CalendarService
from rag.config import settings

async def get_patient_info(
    patient_id: str,
    question: str,
    limit: int = 5,
    include_sensitive: bool = False,
    emotion_filter: Optional[str] = None
) -> str:
    """
    Sends a patient query request to the Fortif.ai RAG API endpoint 
    (POST /api/query).
    
    Args:
        patient_id: Unique identifier for the patient (e.g., "patient_123").
        query_command: The question or prompt for the LLM.
        limit: Max number of documents to retrieve (default is 3).
        include_sensitive: Whether to include sensitive memories.
        emotion_filter: Filter results by emotion (e.g., "positive").

    Returns:
        The generated empathetic response text from the LLM.
    """
    url = settings.api_url + "/api/query"
    print(f"Call URL: with command {question} {url}")

    payload = {
        "patient_id": patient_id,
        "question": question,
        "limit": limit,
        "include_sensitive": include_sensitive,
        "emotion_filter": emotion_filter
    }

    headers = {
        "Content-Type": "application/json",
        "X-API-Key": settings.api_key
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            print(f"Sending RAG query for patient {patient_id}...")
            
            response = await client.post(
                url,
                headers=headers,
                json=payload
            )
            
            # Raises an exception for 4xx or 5xx status codes
            response.raise_for_status() 

            # Return the generated response from the JSON body
            print(response.json().get("response", "Error: 'response' field missing in API result."))
            return response.json().get("response", "Error: 'response' field missing in API result.")

    except httpx.HTTPStatusError as e:
        # Handle API errors (e.g., Invalid API Key, Internal Server Error)
        try:
            error_detail = e.response.json().get("detail", "Unknown API error.")
        except:
            error_detail = e.response.text
        return f"API Error ({e.response.status_code}): {error_detail}"
        
    except httpx.RequestError as e:
        # Handle connection errors (e.g., API server not running)
        return f"Connection Error: Failed to connect to API server. Is the server running? ({e})"

class Processor:
    """Handle different types of LLM actions through a pipeline system."""
    
    def __init__(self):

        #model = whisper.load_model("base")
        # Initialize the LLM (Gemini for testing)
        self.llm = ChatGoogleGenerativeAI(
            model='gemini-2.5-flash',
            api_key=settings.google_api_key #add ur api key here
        )
        
        # Defined actions for the LLM
        #print("we're initialising")
        self.actions = {
            'calendar': CalendarService(self.llm).handle_calendar,
            'pill': get_patient_info,
            'call': get_patient_info,
            'personal': get_patient_info,
        }
    
    async def process_command(self, command, patient_id: str) -> Dict[str, Any]:
        """Process a voice command through the LLM action pipeline."""
        try:
            # Determine the action type from the command
            action_type = await self._determine_action_type(command)
            print("Action Type: " + action_type)
            
            # Get the function for the determined action type
            handler = self.actions.get(action_type)
            print("Handler: " + str(handler))
            
            # Check if handler is defined
            if not handler:
                return {
                    "status": "error",
                    "message": f"No handler for action type: {action_type}"
                }
                
            # Call the handler
            if (action_type == 'calendar'):
                return await handler(command)
            else:
                return await handler(patient_id, command)
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Processing error: {str(e)}"
            }
    
    async def _determine_action_type(self, command: str) -> str:
        """Determine action type from command text"""

        print("Printing the command" + str(command))
        
        prompt = f"""
        Determine the action type based on the command: "{command}"
        Possible action types are: calendar, pill, call, personal info request (personal). 
        Reply with a one word action type in lowercase.
        """
        try:
            # Call the LLM to determine the correct action type
            response = await self.llm.ainvoke(prompt)
            print(response.content.strip())
            return response.content.strip()
        except Exception as e:
            return {
                "status": "error",
                "message": f"Processing error: {str(e)}"
            }