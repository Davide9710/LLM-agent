from BaseLLMEngine import BaseLLMEngine
from typing import List, Dict

class AIAgent:
    def __init__(self, engine: BaseLLMEngine):
        self.engine = engine

    def chat(self, messages: List[Dict[str, str]]) -> str:
        return self.engine(messages)

    def set_engine(self, engine: BaseLLMEngine):
        self.engine = engine