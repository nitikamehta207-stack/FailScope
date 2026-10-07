"""
validator.py - rejects hypotheses that are not backed by real, relevant evidence.

This is the safety net. A hypothesis without proper evidence is thrown away,
no matter how confident it sounds. It matters most once an LLM writes the
hypotheses, because an LLM can invent evidence ids.
"""
from dataclasses import dataclass

# Which tools can say something meaningful about each cause
RELEVANT_TOOLS = {
    "bad_deploy": {"get_deploy_history", "evaluate_model_version", "get_metric_series",
                   "compare_distributions", "check_data_quality", "read_logs"},
    "data_drift": {"compare_distributions", "get_metric_series", "check_data_quality",
                   "get_deploy_history", "evaluate_model_version"},
    "schema_change": {"get_deploy_history", "check_data_quality", "get_metric_series",
                      "read_logs"},
}


@dataclass
class ValidationResult:
    hypothesis: object
    valid: bool
    reasons: list      # why it was rejected (empty if valid)


def validate(hypothesis, evidence, min_evidence=2, min_layers=2):
    reasons = []
    if hypothesis.cause not in RELEVANT_TOOLS:
        reasons.append(f"unknown cause '{hypothesis.cause}'")
    if not 0 <= hypothesis.confidence <= 1:
        reasons.append("confidence is not between 0 and 1")

    cited = []
    for eid in hypothesis.evidence_ids:
        if eid not in evidence:
            reasons.append(f"evidence id {eid} does not exist")
        else:
            cited.append(evidence[eid])

    relevant = RELEVANT_TOOLS.get(hypothesis.cause, set())
    for e in cited:
        if e.tool not in relevant:
            reasons.append(f"{e.id} ({e.tool}) is not relevant to {hypothesis.cause}")

    good = [e for e in cited if e.tool in relevant]
    if len(good) < min_evidence:
        reasons.append(f"needs at least {min_evidence} relevant evidence items, has {len(good)}")
    if good and not any(e.anomalous for e in good):
        reasons.append("none of the cited evidence shows an anomaly")
    if len({e.layer for e in good}) < min_layers:
        reasons.append(f"evidence must come from at least {min_layers} layers (data/model/system)")
    return ValidationResult(hypothesis, not reasons, reasons)