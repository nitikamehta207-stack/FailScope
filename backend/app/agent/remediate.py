"""
remediate.py - closes the loop: apply the fix, then VERIFY the system recovered.

Flow (run_incident):
  1. The agent investigates what happened up to fix_day.
  2. For the best hypothesis: ask for approval, apply its action in the sandbox.
  3. verify_recovery compares the days after the fix with the healthy baseline.
  4. If it did not recover, undo that fix and try the next hypothesis (max 2).
  5. If nothing works, escalate to a human.

Each attempt starts from the same state (a failed fix is undone first).
"""
from dataclasses import dataclass, field

from .agent import Investigator
from .tools import Toolbox


@dataclass
class Verification:
    recovered: bool
    details: dict
    summary: str


@dataclass
class Attempt:
    hypothesis: object
    verification: Verification


@dataclass
class IncidentResult:
    report: object
    attempts: list = field(default_factory=list)
    resolved: bool = False
    log: list = field(default_factory=list)


def verify_recovery(store_after, fix_day, baseline=(0, 9), tolerance=0.04):
    """Healthy again = accuracy and AUC back near baseline AND inputs not broken."""
    m = store_after.metrics()
    base = m[(m.day >= baseline[0]) & (m.day <= baseline[1])]
    before = m[(m.day >= fix_day - 5) & (m.day < fix_day)]
    after = m[m.day >= fix_day]
    last_day = int(m.day.max())

    acc_base, auc_base = base.accuracy.mean(), base.auc.mean()
    acc_before, acc_after, auc_after = before.accuracy.mean(), after.accuracy.mean(), after.auc.mean()

    # Are the inputs themselves still broken (null spike, wrong units)?
    quality = Toolbox(store_after, baseline=baseline, recent=(fix_day, last_day)).check_data_quality()

    acc_ok = acc_after >= acc_base - tolerance
    auc_ok = auc_after >= auc_base - tolerance
    data_ok = not quality.anomalous
    details = {"baseline_accuracy": round(float(acc_base), 3),
               "accuracy_before_fix": round(float(acc_before), 3),
               "accuracy_after_fix": round(float(acc_after), 3),
               "baseline_auc": round(float(auc_base), 3),
               "auc_after_fix": round(float(auc_after), 3),
               "accuracy_ok": bool(acc_ok), "auc_ok": bool(auc_ok), "data_ok": bool(data_ok),
               "data_quality": quality.finding}
    problems = []
    if not acc_ok:
        problems.append(f"accuracy still low ({acc_after:.3f} vs baseline {acc_base:.3f})")
    if not auc_ok:
        problems.append(f"AUC still low ({auc_after:.3f} vs baseline {auc_base:.3f})")
    if not data_ok:
        problems.append("inputs are still broken: " + quality.finding.rstrip("."))
    recovered = acc_ok and auc_ok and data_ok
    summary = (f"RECOVERED: accuracy {acc_before:.3f} -> {acc_after:.3f} (baseline {acc_base:.3f}), "
               f"AUC {auc_after:.3f}, inputs healthy." if recovered
               else "NOT RECOVERED: " + "; ".join(problems) + ".")
    return Verification(recovered, details, summary)


def run_incident(sim, fault, seed=0, variant="unit", observe_days=23, total_days=30,
                 investigator=None, approve=None, max_attempts=2, hypotheses=None):
    """
    observe_days : the agent sees days 0..observe_days-1, the fix is applied on day observe_days
    approve      : optional function(hypothesis) -> bool (human approval). None = auto-approve
    hypotheses   : optional list to try instead of the agent's own (used to demo a wrong first guess)
    """
    investigator = investigator or Investigator()
    before = sim.run(n_days=observe_days, fault=fault, seed=seed, variant=variant)
    report = investigator.investigate(before)
    result = IncidentResult(report)

    if not report.incident:
        result.log.append("No incident detected, nothing to fix.")
        return result

    candidates = hypotheses if hypotheses is not None else report.valid
    if not candidates:
        result.log.append("No validated hypothesis. Escalating to a human.")
        return result

    for i, h in enumerate(candidates[:max_attempts], start=1):
        result.log.append(f"Attempt {i}: hypothesis '{h.cause}' -> action '{h.action}'")
        if approve is not None and not approve(h):
            result.log.append("  Human declined this action. Stopping.")
            return result
        after = sim.run(n_days=total_days, fault=fault, seed=seed, variant=variant,
                        fix=h.action, fix_day=observe_days)
        v = verify_recovery(after, observe_days)
        result.attempts.append(Attempt(h, v))
        result.log.append("  " + v.summary)
        if v.recovered:
            result.resolved = True
            return result
        result.log.append("  Undoing this fix before trying the next hypothesis.")

    result.log.append("Could not recover the system automatically. Escalating to a human.")
    return result


def format_incident(r):
    lines = [f"  Agent found: {r.report.top.cause if r.report.top else 'nothing'}"
             + (f" (confidence {r.report.top.confidence:.2f})" if r.report.top else "")]
    lines += ["  " + x for x in r.log]
    lines.append("  FINAL: " + ("system recovered." if r.resolved else "NOT resolved, needs a human."))
    return "\n".join(lines)