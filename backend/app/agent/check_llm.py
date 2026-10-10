"""
check_llm.py - one tiny call to confirm the API key and model work.
Run from the FailScope folder:  python -m backend.app.agent.check_llm
"""
from .llm_client import LLMClient


def main():
    llm = LLMClient()
    print("model:", llm.model)
    reply = llm.complete_json("You reply only with JSON.",
                              'Return this JSON: {"ok": true, "word": "hello"}')
    print("reply:", reply)


if __name__ == "__main__":
    main()