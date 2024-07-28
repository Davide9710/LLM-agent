from abc import ABC, abstractmethod
from typing import List, Dict

class BaseLLMEngine(ABC):
    @abstractmethod
    def __call__(self, messages: List[Dict[str, str]], stop_sequences: List[str] = []) -> str:
        pass