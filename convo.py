from simple_llm.clients.openai import OpenAIAgent
from typing import Generator

class LLMService:
    def __init__(self, full_system_prompt: str):
        # Create a single agent directly, using the full prompt as the system message
        self.agent = OpenAIAgent(
            name="assistant",
            model="gpt-4o-mini",
            system_message=full_system_prompt, # Combined RAG context
            stream=True
        )

    def stream_reply(self, query: str) -> str | Generator:
        try:
            for chunk in self.agent.stream_reply(query):
                yield chunk
        except Exception as e:
            # Handle API/streaming errors here
            print(f"LLM API Error: {e}")
            yield "Sorry, I ran into an error generating a response."