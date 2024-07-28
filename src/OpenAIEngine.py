import openai
import os
from dotenv import load_dotenv
from typing import List, Dict
from BaseLLMEngine import BaseLLMEngine

load_dotenv()

class OpenAIEngine(BaseLLMEngine):
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: str = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set it as an argument or as OPENAI_API_KEY environment variable.")
        openai.api_key = self.api_key
        self.model = model

    def __call__(self, messages: List[Dict[str, str]], stop_sequences: List[str] = []) -> str:
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            stop=stop_sequences,
            max_tokens=100
        )
        return response.choices[0].message.content