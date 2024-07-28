from AIAgent import AIAgent
from HfEngine import HfEngine
from OpenAIEngine import OpenAIEngine
from ClaudeEngine import ClaudeEngine

def main():
    # Initialize with HuggingFace
    hf_engine = HfEngine()
    agent = AIAgent(hf_engine)

    # Chat using HuggingFace
    response = agent.chat([{"role": "user", "content": "Hello, AI!"}])
    print(response)

    # Switch to OpenAI
    openai_engine = OpenAIEngine()
    agent.set_engine(openai_engine)

    # Chat using OpenAI
    response = agent.chat([{"role": "user", "content": "Hello again, AI!"}])
    print(response)

    # Switch to Claude
    claude_engine = ClaudeEngine()
    agent.set_engine(claude_engine)

    # Chat using Claude
    response = agent.chat([{"role": "user", "content": "Hello once more, AI!"}])
    print(response)