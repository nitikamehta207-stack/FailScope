"""
llm_reasoner.py - an LLM reads the Evidence and writes the hypotheses.

Same interface as RuleReasoner:  propose(evidence) -> list[Hypothesis]

Safety design:
  * the LLM writes cause, confidence, evidence_ids and reasoning only
  * the ACTION always comes from our own code (ACTION_FOR), never from the LLM
  * everything the LLM says still goes through the validator afterwards
  * if the API fails or the JSON is unusable, we fall back to RuleReasoner
"""
from .hypotheses import Hypothesis, ACTION_FOR
from .reasoner import RuleReasoner

SYSTEM_PROMPT = """You are an ML reliability investigator. A machine-learning service (a loan default model) started performing badly. You receive evidence items produced by deterministic tools. Each has an id like E4.

Decide the root cause. The candidate causes are exactly these three:
- data_drift: the real world changed gradually (for example customers earn less). The inputs are valid but different, and the true outcomes shift too. Typical fix: retrain on recent data.
- schema_change: a data pipeline change broke the inputs (wrong units, missing values) while the real world did not change. Typical fix: revert the pipeline change.
- bad_deploy: a newly deployed model version is worse than the previous one. Typical fix: roll back the model.

Rules:
- Use ONLY evidence ids that appear in the evidence list. Never invent an id or a fact.
- Cite at least two evidence ids per hypothesis, from at least two different layers (data, model, system).
- Evidence that shows something is normal can also support a hypothesis (for example the old model still works).
- Return all three causes, ranked from most to least likely.
- confidence is a number from 0 to 1: how well the evidence supports that cause.
- Reply with JSON only."""

USER_TEMPLATE = """Evidence:
{evidence}

Reply with JSON in exactly this shape:
{{"hypotheses": [{{"cause": "data_drift | schema_change | bad_deploy", "confidence": 0.0, "evidence_ids": ["E1", "E2"], "reasoning": "short explanation that mentions the evidence ids"}}]}}"""


def format_evidence(evidence):
    lines = []
    for e in evidence.values():
        flag = "yes" if e.anomalous else "no"
        lines.append(f"[{e.id}] layer={e.layer} tool={e.tool} anomalous={flag}: {e.finding}")
    return "\n".join(lines)


class LLMReasoner:
    def __init__(self, client=None, fallback=None):
        if client is None:
            from .llm_client import LLMClient      # imported here so tests need no API key
            client = LLMClient()
        self.client = client
        self.fallback = fallback or RuleReasoner()
        self.last_source = "not run yet"           # "llm" or "rules (...)"
        self.last_error = ""                       # the real error text if the LLM failed

    def propose(self, evidence):
        user = USER_TEMPLATE.format(evidence=format_evidence(evidence))
        try:
            data = self.client.complete_json(SYSTEM_PROMPT, user)
            hypotheses = self._parse(data)
            if not hypotheses:
                raise ValueError("LLM returned no hypotheses")
            self.last_source, self.last_error = "llm", ""
            return hypotheses
        except Exception as err:
            self.last_source = f"rules (LLM failed: {type(err).__name__})"
            self.last_error = str(err)
            return self.fallback.propose(evidence)

    @staticmethod
    def _parse(data):
        out = []
        for item in data.get("hypotheses", []):
            cause = str(item.get("cause", "")).strip()
            try:
                confidence = min(1.0, max(0.0, float(item.get("confidence", 0))))
            except (TypeError, ValueError):
                confidence = 0.0
            ids = [str(x).strip() for x in item.get("evidence_ids", [])]
            out.append(Hypothesis(cause=cause, confidence=round(confidence, 2),
                                  evidence_ids=ids, reasoning=str(item.get("reasoning", "")),
                                  action=ACTION_FOR.get(cause, "")))     # action comes from OUR code
        out.sort(key=lambda h: h.confidence, reverse=True)
        return out