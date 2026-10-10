"""
test_llm_reasoner.py - checks that the validator catches a misbehaving LLM.
It uses a FAKE LLM, so it needs no API key and no internet.
Run from the FailScope folder:  python -m backend.app.agent.test_llm_reasoner
"""
import re

from ..sim.simulator import Simulator
from .agent import Investigator
from .llm_reasoner import LLMReasoner


def ids_for(prompt, tool):
    """Find the evidence ids in the prompt that came from a given tool."""
    return re.findall(r"\[(E\d+)\] layer=\w+ tool=" + tool + r" ", prompt)


class FakeLLM:
    """Pretends to be the LLM. `behaviour` decides how it misbehaves."""
    def __init__(self, behaviour):
        self.behaviour = behaviour

    def complete_json(self, system, user):
        b = self.behaviour
        deploy = ids_for(user, "get_deploy_history")
        evals = ids_for(user, "evaluate_model_version")
        dist = ids_for(user, "compare_distributions")

        def hyp(cause, conf, ids, why="fake reasoning"):
            return {"cause": cause, "confidence": conf, "evidence_ids": ids, "reasoning": why}

        if b == "good":
            return {"hypotheses": [hyp("bad_deploy", 0.9, deploy + evals[:1] + dist),
                                   hyp("schema_change", 0.05, deploy),
                                   hyp("data_drift", 0.05, dist)]}
        if b == "invented_ids":
            return {"hypotheses": [hyp("bad_deploy", 0.95, ["E91", "E92", "E93"])]}
        if b == "irrelevant_evidence":
            return {"hypotheses": [hyp("schema_change", 0.9, dist + evals[:1])]}
        if b == "one_layer_only":
            return {"hypotheses": [hyp("bad_deploy", 0.9, evals)]}
        if b == "unknown_cause":
            return {"hypotheses": [hyp("solar_flare", 0.99, deploy + evals[:1])]}
        if b == "garbage":
            raise ValueError("model returned text that is not JSON")
        raise AssertionError(b)


CASES = [
    # (behaviour, what we expect the final answer to be)
    ("good", "bad_deploy"),
    ("invented_ids", "none"),
    ("irrelevant_evidence", "none"),
    ("one_layer_only", "none"),
    ("unknown_cause", "none"),
    ("garbage", "bad_deploy"),          # LLM fails -> falls back to the rule reasoner
]


def main():
    sim = Simulator()
    store = sim.run(n_days=23, fault="bad_deploy", seed=0)    # the real fault is bad_deploy
    passed = 0
    for behaviour, expected in CASES:
        reasoner = LLMReasoner(client=FakeLLM(behaviour))
        report = Investigator(reasoner=reasoner).investigate(store)
        got = report.top.cause if report.top else "none"
        ok = got == expected
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  fake LLM behaviour = {behaviour:<20} "
              f"source = {reasoner.last_source:<32} final = {got}")
        for r in report.rejected:
            if r.reasons and r.hypothesis.confidence >= 0.5:
                print(f"        rejected '{r.hypothesis.cause}': {'; '.join(r.reasons)}")
    print(f"\n{passed}/{len(CASES)} checks passed")


if __name__ == "__main__":
    main()