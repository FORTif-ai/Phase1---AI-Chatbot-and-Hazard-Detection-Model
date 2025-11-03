from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
import base64
from langchain_core.messages import HumanMessage
import whisper

# Import services
from mycalendar import CalendarService
from maindelegator import delegation

load_dotenv("rag/.env")

class Processor:
    """Handle different types of LLM actions through a pipeline system."""
    
    def __init__(self):

        #model = whisper.load_model("base")
        # Initialize the LLM (Gemini for testing)
        self.llm = ChatGoogleGenerativeAI(
            model='gemini-2.5-flash',
            api_key=os.getenv("GOOGLE_API_KEY")  # Use the API key from the environment variable
        )
        
        # Defined actions for the LLM
        #print("we're initialising")
        self.actions = {
            'calendar': CalendarService(self.llm).handle_calendar,
            'pill': delegation.delegate,
            'call': delegation.delegate,
            'personal': delegation.delegate
        }
    
    async def process_audio(self, audio: bytes, mime_type: str = 'audio/wav') -> Dict[str, Any]:
        model = whisper.load_model("base")
        """Transcribe raw audio to text and process it through the LLM action pipeline."""
        try:
            #print("srsly bruhhhhhhhhhh")
            # Convert audio to base64
            #audio_base64 = base64.b64encode(audio).decode('utf-8')
            #transcript="Appointment for tomorrow"
            #print(f"Transcript: {transcript}")
            # Create a message with the audio
            """
            message = HumanMessage(
                content=[
                    {"type": "text",  "text": "Transcribe the audio and if you can't just output the following: I would like an appointment for tomorrow"},
                    {"type": "media", "data": audio, "mime_type": mime_type},
                ]
            )
            
            # Get the transcript from Gemini
            response = await self.llm.acall([message])
            transcript = response.content.strip()
            print("We've go a transcript yall")
            print(f"Transcript: {transcript}")
        
            """
            result = model.transcribe(audio)
            print("Transcription:", result["text"])
            transcript = result["text"]
            
            # Route the transcript to the appropriate handler
            return await self.process_command(transcript)
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Processing error: {str(e)}"
            }
    
    async def process_command(self, command: str) -> Dict[str, Any]:
        """Process a voice command through the LLM action pipeline."""
        try:
            # Determine the action type from the command
            action_type = await self._determine_action_type(command)
            print("Action Type: " + action_type)
            
            # Get the function for the determined action type
            handler = self.actions.get(action_type)
            
            # Check if handler is defined
            if not handler:
                return {
                    "status": "error",
                    "message": f"No handler for action type: {action_type}"
                }
                
            # Call the handler
            return await handler(command)
            
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
        

