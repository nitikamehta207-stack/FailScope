"""
agent.py - the investigator: triage -> gather evidence -> hypotheses -> validate.

The agent only sees the Toolbox (telemetry through tools). It never sees
ground_truth.
"""
from dataclasses import dataclass, field

from ..sim.dataset import FEATURES
from .tools import Toolbox
from .reasoner import RuleReasoner
from .validator import validate

MIN_CONFIDENCE = 0.4     # below this we do not trust a hypothesis enough to act on it


@dataclass
class Report:
    incident: bool
    change_day: int = None
    toolbox: object = None
    valid: list = field(default_factory=list)       # validated hypotheses, best first
    rejected: list = field(default_factory=list)    # ValidationResult objects
    top: object = None                              # best valid hypothesis (or None)
    layers_checked: list = field(default_factory=list)
    trace: list = field(default_factory=list)       # step-by-step log of what the agent did


class Investigator:
    def __init__(self, reasoner=None):
        self.reasoner = reasoner or RuleReasoner()

    def investigate(self, store):
        tb = Toolbox(store)
        trace = []

        # ---- 1. Triage: scan the key signals, find when things changed ----
        scan = ["accuracy", "auc", "actual_default_rate", "max_null_rate"] + \
               [f"mean_{f}" for f in FEATURES]
        found = [tb.get_metric_series(m) for m in scan]
        flagged = [e for e in found if e.anomalous]
        trace.append(f"Triage: scanned {len(scan)} signals, {len(flagged)} look abnormal.")
        if not flagged:
            trace.append("No incident detected.")
            return Report(False, None, tb, trace=trace)
        change_day = min(e.data["change_day"] for e in flagged)
        trace.append(f"Problem seems to start on day {change_day}.")

        # ---- 2. Gather evidence from all three layers ----
        tb.compare_distributions()
        tb.check_data_quality()
        trace.append("Data layer: compared distributions and checked data quality.")
        tb.get_deploy_history(around_day=change_day)
        tb.read_logs(day_from=max(change_day - 1, 0), day_to=change_day + 1)
        trace.append(f"System layer: checked deploys and logs around day {change_day}.")
        versions = list(dict.fromkeys([tb.live_version] +
                        list(tb.deploys_df[tb.deploys_df.component == "model"].version)))
        for v in versions:
            tb.evaluate_model_version(v)
        trace.append(f"Model layer: replayed model versions {versions} on recent traffic.")
        layers = sorted({e.layer for e in tb.evidence.values()})
        trace.append(f"Layers checked: {layers}.")

        # ---- 3. Hypotheses, then validation ----
        proposed = self.reasoner.propose(tb.evidence)
        source = getattr(self.reasoner, "last_source", "rules")
        trace.append(f"Reasoner ({source}) proposed {len(proposed)} hypotheses.")
        valid, rejected = [], []
        for h in proposed:
            result = validate(h, tb.evidence)
            if result.valid and h.confidence >= MIN_CONFIDENCE:
                valid.append(h)
            else:
                rejected.append(result)
        trace.append(f"Validator kept {len(valid)}, rejected {len(rejected)}.")
        top = valid[0] if valid else None
        return Report(True, change_day, tb, valid, rejected, top, layers, trace)


def format_report(r):
    lines = []
    for t in r.trace:
        lines.append("  - " + t)
    if r.top:
        lines.append(f"  ROOT CAUSE: {r.top.cause} (confidence {r.top.confidence:.2f}) "
                     f"-> action: {r.top.action}")
        lines.append(f"  Evidence: {', '.join(r.top.evidence_ids)}")
        lines.append(f"  Why: {r.top.reasoning}")
    elif r.incident:
        lines.append("  No hypothesis passed validation. Escalate to a human.")
    return "\n".join(lines)