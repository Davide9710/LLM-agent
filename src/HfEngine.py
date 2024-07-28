from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv
from typing import List, Dict
from BaseLLMEngine import BaseLLMEngine

load_dotenv()

class HfEngine(BaseLLMEngine):
    def __init__(self, model: str = "gpt2", api_token: str = None):
        self.api_token = api_token or os.environ.get("HF_API_TOKEN")
        if not self.api_token:
            raise ValueError("HuggingFace API token not provided. Set it as an argument or as HF_API_TOKEN environment variable.")
        self.client = InferenceClient(model=model, token=self.api_token)

    def __call__(self, messages: List[Dict[str, str]], stop_sequences: List[str] = []) -> str:
        prompt = " ".join([f"{m['role']}: {m['content']}" for m in messages])
        response = self.client.text_generation(prompt, stop_sequences=stop_sequences, max_new_tokens=100)
        return response