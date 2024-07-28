from anthropic import Anthropic
import os   
from dotenv import load_dotenv
from typing import List, Dict
from BaseLLMEngine import BaseLLMEngine

load_dotenv()

class ClaudeEngine(BaseLLMEngine):
    def __init__(self, model: str = "claude-2.1", api_key: str = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided. Set it as an argument or as ANTHROPIC_API_KEY environment variable.")
        self.client = Anthropic(api_key=self.api_key)
        self.model = model

    def __call__(self, messages: List[Dict[str, str]], stop_sequences: List[str] = []) -> str:
        response = self.client.messages.create(
            model=self.model,
            messages=messages,
            stop_sequences=stop_sequences,
            max_tokens=100
        )
        return response.content[0].text