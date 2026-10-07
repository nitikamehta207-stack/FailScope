"""
hypotheses.py - what the agent produces: a ranked guess about the root cause.
"""
from dataclasses import dataclass, field, asdict

# The corrective action that goes with each cause (executed in the next step)
ACTION_FOR = {"bad_deploy": "rollback_model",
              "schema_change": "revert_pipeline_change",
              "data_drift": "retrain_on_recent"}


@dataclass
class Hypothesis:
    cause: str                 # "bad_deploy" | "schema_change" | "data_drift"
    confidence: float          # 0..1 (share of the expected clues that were found)
    evidence_ids: list = field(default_factory=list)   # ["E4", "E8", ...]
    reasoning: str = ""        # plain-English explanation that cites evidence ids
    action: str = ""           # recommended fix

    def to_dict(self):
        return asdict(self)