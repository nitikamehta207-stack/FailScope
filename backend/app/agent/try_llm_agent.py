"""
try_llm_agent.py - the full loop with the REAL LLM: investigate -> fix -> verify.
Needs the API key in .env. Makes 3 LLM calls (one per fault).
Run from the FailScope folder:  python -m backend.app.agent.try_llm_agent
"""
from ..sim.simulator import Simulator
from .agent import Investigator
from .llm_reasoner import LLMReasoner
from .remediate import run_incident, format_incident

FAULTS = ["data_drift", "schema_change", "bad_deploy"]


def main():
    sim = Simulator()
    reasoner = LLMReasoner()
    agent = Investigator(reasoner=reasoner)
    for fault in FAULTS:
        print("=" * 90)
        print("FAULT (hidden from agent):", fault)
        result = run_incident(sim, fault, seed=0, investigator=agent)
        print(format_incident(result))
        print("  Reasoner used:", reasoner.last_source)
        if reasoner.last_error:
            print("  LLM error:", reasoner.last_error[:300])
        if result.report.top:
            print("  LLM reasoning:", result.report.top.reasoning)
        for rej in result.report.rejected:
            if rej.reasons:
                print(f"  Rejected '{rej.hypothesis.cause}' (conf {rej.hypothesis.confidence}): "
                      + "; ".join(rej.reasons))


if __name__ == "__main__":
    main()