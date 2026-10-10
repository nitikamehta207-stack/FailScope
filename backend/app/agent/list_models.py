"""
list_models.py - prints the model names your Groq key can use.
Run from the FailScope folder:  python -m backend.app.agent.list_models
"""
from .llm_client import LLMClient


def main():
    llm = LLMClient()
    names = sorted(m.id for m in llm.client.models.list().data)
    for name in names:
        print(name)


if __name__ == "__main__":
    main()