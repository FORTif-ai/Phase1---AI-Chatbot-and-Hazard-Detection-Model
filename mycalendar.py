from typing import Dict, Any
import os
from datetime import datetime
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from langchain_google_genai import ChatGoogleGenerativeAI
import re
import json
from dotenv import load_dotenv
load_dotenv()

class CalendarService:
    def __init__(self, llm: ChatGoogleGenerativeAI):
        """Initialize the Calendar Service with optional LLM for parsing commands."""
        self.llm = llm
        self.calendar_service = self._init_calendar_service()

    def _init_calendar_service(self) -> Any:
        """Initialize the Google Calendar API service."""
        try:
            creds = None
            #token_path = os.getenv("D:\\chatbot_merged\\token.json")
            token_path = "D:\\chatbot_merged\\token.json" #update file path


            
            # Validate the credentials
            if os.path.exists(token_path):
                creds = Credentials.from_authorized_user_info(
                    info=eval(open(token_path, 'r').read()),
                    scopes=['https://www.googleapis.com/auth/calendar']
                )
            
            if not creds or not creds.valid:
                raise ValueError("Invalid or missing credentials")
                
            return build('calendar', 'v3', credentials=creds)
        except Exception as e:
            print(f"Failed to initialize calendar service for real: {e}")
            return None
    
    async def _parse_calendar_command(self, command: str) -> Dict[str, Any]:
        """Parse a natural language calendar command into structured event data."""
        # Check if LLM is initialized
        if not self.llm:
            raise ValueError("LLM not initialized. Cannot parse natural language commands.")

        # Get current time
        current_time = datetime.now()
        
        # Define the prompt for parsing the command
        prompt = f"""
        Current date and time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} Eastern Time (UTC-4)
        
        Parse the following calendar command into structured event data:
        "{command}"
        
        If you see a [follow-up for missing fields] in the command, use it as part of the event data. If there's time information in the follow-up without a AM/PM, use logic to determine if it's AM or PM.   
        
        Extract these fields:
        - title: The title or subject of the event
        - start_time: The start date and time in Eastern Time (UTC-4) (format as ISO string with timezone offset)
        - end_time: The end date and time in Eastern Time (UTC-4) (format as ISO string with timezone offset)
        - description: Any additional details about the event
        - location: The location of the event, if specified
        - attendees: List of email addresses for attendees, if specified
        
        DO NOT MAKE UP ANY DETAILS THAT ARE NOT IN THE COMMAND, LEAVE THE FIELD OUT IF IT IS NOT IN THE COMMAND. 
        
        For relative time expressions like "tomorrow" or "next week", use the current date/time as reference.
        Always return dates in ISO format with timezone offset (YYYY-MM-DDTHH:MM:SS-04:00).
        All times should be in Eastern Time (UTC-4).
        
        Assume events without a specified year are in the current year {current_time.year} and events without a specified time are in the current day {current_time.strftime('%Y-%m-%d')}.
        
        Return only a JSON object with these fields, nothing else.
        """
        
        try:
            response = await self.llm.ainvoke(prompt)
            content = response.content
            
            # Fallback parsing in case LLM doesn't return proper JSON
            if not isinstance(content, dict):
                # Regex parsing as fallback
                title_match = re.search(r"title[\"']?\s*:\s*[\"']([^\"']+)[\"']", content)
                start_match = re.search(r"start_time[\"']?\s*:\s*[\"']([^\"']+)[\"']", content)
                end_match = re.search(r"end_time[\"']?\s*:\s*[\"']([^\"']+)[\"']", content)
                desc_match = re.search(r"description[\"']?\s*:\s*[\"']([^\"']+)[\"']", content)
                loc_match = re.search(r"location[\"']?\s*:\s*[\"']([^\"']+)[\"']", content)
                
                event_data = {}
                if title_match:
                    event_data["title"] = title_match.group(1)
                if start_match:
                    event_data["start_time"] = start_match.group(1)
                if end_match:
                    event_data["end_time"] = end_match.group(1)
                if desc_match:
                    event_data["description"] = desc_match.group(1)
                if loc_match:
                    event_data["location"] = loc_match.group(1)
                event_data["attendees"] = []
            else:
                event_data = content
            
            # Validate event times
            self._validate_event_times(event_data)
            
            return event_data
        except Exception as e:
            raise ValueError(f"Failed to parse event: {str(e)}")
    
    def _validate_event_times(self, event_data: Dict[str, Any]):
        """Validate event times that are present in the event data."""
        try:
            # Only validate start_time if it exists
            if 'start_time' in event_data:
                start_time = datetime.fromisoformat(event_data['start_time'].replace('Z', '+00:00'))
                current_time = datetime.now(start_time.tzinfo)
                
                # Check if start time is in the past
                if start_time < current_time:
                    raise ValueError(f"Cannot create events in the past. Event start time {start_time} is before current time {current_time}")
                
                # Only validate end_time if both times exist
                if 'end_time' in event_data:
                    end_time = datetime.fromisoformat(event_data['end_time'].replace('Z', '+00:00'))
                    if end_time < start_time:
                        raise ValueError(f"Event end time {end_time} cannot be before start time {start_time}")
            
            return event_data
        except Exception as e:
            raise ValueError(f"Invalid event times: {str(e)}")
    
    def _save_to_log(self, event: Dict[str, Any]):
        """Save essential event details to a JSON file for model processing."""
        try:
            file_path = "calendar_events.json"
            events = []
            
            # Load existing events if file exists
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        events = json.load(f)
                except json.JSONDecodeError:
                    # If file is corrupted, start with empty list
                    events = []
            
            # Extract relevant information
            saved_event = {
                "id": event.get("id"),
                "summary": event.get("summary"),
                "status": event.get("status"),
                "start": event.get("start", {}).get("dateTime"),
                "end": event.get("end", {}).get("dateTime"),
                "link": event.get("htmlLink")
            }
            
            # Add new event
            events.append(saved_event)
            
            # Save updated events list
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(events, indent=2, fp=f)
        except Exception as e:
            print(f"Warning: Failed to log event to file: {str(e)}")

    async def handle_calendar(self, command: str) -> Dict[str, Any]:
        """Handle calendar-related commands."""
        # Check if calendar service is initialized
        if not self.calendar_service:
            return {
                "status": "error",
                "action_type": "calendar",
                "error_message": "Calendar service not initialized. Check Google Calendar API credentials."
            }
        
        try:
            # Parse the command into event details
            event_data = await self._parse_calendar_command(command)
            
            # Reprompt for missing required fields
            required_fields = ["title", "start_time", "end_time"]
            
            def get_missing_fields():
                missing = []
                for field in required_fields:
                    if not event_data.get(field):
                        missing.append(field)
                return missing
            
            missing = get_missing_fields()
            if missing:
                # Display more readable field names
                display_fields = [field.replace('_', ' ') for field in missing]
                prompt_str = f"I didn't catch your {', '.join(display_fields)}. Could you tell me those details? "
                # Wait for user input in CLI
                user_response = input(prompt_str)
                # Provide the command and user update to the parser
                followup = (command + " [follow-up for missing fields " + str(missing) + "]: " + user_response)
                new_data = await self._parse_calendar_command(followup)
                # Update the event data with the new data
                for key in missing:
                    if new_data.get(key):
                        event_data[key] = new_data[key]
                # Re-validate event times if updated
                self._validate_event_times(event_data)
                missing = get_missing_fields()
                if missing:
                    raise ValueError(f"Missing required fields after follow-up: {missing}")
            
            # Format the event for Google Calendar API
            event = {
                'summary': event_data.get('title', 'Untitled Event'),
                'description': event_data.get('description', ''),
                'location': event_data.get('location', ''),
                'start': {
                    'dateTime': event_data.get('start_time'),
                    'timeZone': 'UTC',
                },
                'end': {
                    'dateTime': event_data.get('end_time'),
                    'timeZone': 'UTC',
                }
            }
            
            # Add attendees if specified
            if event_data.get('attendees'):
                event['attendees'] = [{'email': email} for email in event_data['attendees']]
            
            # Create the event in the primary calendar
            created_event = self.calendar_service.events().insert(
                calendarId='primary',
                body=event
            ).execute()
            
            # Save the event to a log file
            self._save_to_log(created_event)
            
            return {
                "status": "success",
                "action_type": "calendar",
                "data": {
                    "event_id": created_event.get('id'),
                    "event_link": created_event.get('htmlLink'),
                    "summary": event_data.get('title'),
                    "start_time": event_data.get('start_time'),
                    "end_time": event_data.get('end_time'),
                    "confirmation": "Event successfully added to Google Calendar"
                }
            }
        
        except Exception as e:
            return {
                "status": "error",
                "action_type": "calendar",
                "error_message": f"Failed to create calendar event: {str(e)}",
                "command": command
            }