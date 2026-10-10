"""
demo.py - the logic behind the Streamlit demo (no Streamlit code in here).

Keeping it separate means we can test it without a browser.

Flow of one incident:
  start_incident -> the agent investigates days 0..22 and proposes fixes
  apply_fix      -> a human approved: apply the next proposed fix, then verify recovery
  decline        -> a human said no: stop
If a fix fails verification, the next proposed hypothesis (max 2) can be approved.
"""
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .agent.agent import Investigator
from .agent.reasoner import RuleReasoner
from .agent.remediate import verify_recovery

OBSERVE_DAYS, TOTAL_DAYS, FAULT_DAY = 23, 30, 15
MAX_ATTEMPTS = 2

# label shown in the dropdown -> (fault, schema_change variant)
SCENARIOS = {
    "Healthy (no fault)": (None, "unit"),
    "Data drift (customers earn less)": ("data_drift", "unit"),
    "Schema change: income unit broken": ("schema_change", "unit"),
    "Schema change: null spike": ("schema_change", "nulls"),
    "Bad model deploy": ("bad_deploy", "unit"),
}


@dataclass
class Attempt:
    hypothesis: object
    after: object            # TelemetryStore of the run with this fix applied
    verification: object


@dataclass
class Incident:
    fault: str
    variant: str
    seed: int
    before: object           # TelemetryStore, days 0..22 (what the agent saw)
    report: object
    queue: list              # hypotheses we may try, best first (max 2)
    reasoner_source: str = "rules"
    attempts: list = field(default_factory=list)
    resolved: bool = False
    declined: bool = False


def make_reasoner(use_llm):
    """Returns (reasoner, note). Never raises: falls back to the rule reasoner."""
    if not use_llm:
        return RuleReasoner(), "Rule-based reasoner (no API calls)."
    try:
        from .agent.llm_reasoner import LLMReasoner
        return LLMReasoner(), "LLM reasoner (falls back to rules if the API fails)."
    except Exception as err:
        return RuleReasoner(), f"LLM unavailable ({type(err).__name__}: {err}). Using the rule-based reasoner."


def start_incident(sim, fault, variant, seed, reasoner):
    before = sim.run(n_days=OBSERVE_DAYS, fault=fault, seed=seed, variant=variant)
    report = Investigator(reasoner).investigate(before)
    source = getattr(reasoner, "last_source", "rules") if report.incident else "not needed"
    return Incident(fault, variant, seed, before, report, list(report.valid[:MAX_ATTEMPTS]), source)


def pending(inc):
    """The next hypothesis waiting for human approval, or None."""
    if inc.resolved or inc.declined or len(inc.attempts) >= len(inc.queue):
        return None
    return inc.queue[len(inc.attempts)]


def apply_fix(sim, inc):
    h = pending(inc)
    if h is None:
        return
    after = sim.run(n_days=TOTAL_DAYS, fault=inc.fault, seed=inc.seed, variant=inc.variant,
                    fix=h.action, fix_day=OBSERVE_DAYS)
    v = verify_recovery(after, OBSERVE_DAYS)
    inc.attempts.append(Attempt(h, after, v))
    inc.resolved = bool(v.recovered)


def decline(inc):
    inc.declined = True


def status(inc):
    """One of: healthy, awaiting_approval, recovered, declined, escalated."""
    if not inc.report.incident:
        return "healthy"
    if inc.resolved:
        return "recovered"
    if inc.declined:
        return "declined"
    if pending(inc) is not None:
        return "awaiting_approval"
    return "escalated"        # no hypothesis left (or none passed validation)


def evidence_by_layer(inc):
    out = {"data": [], "model": [], "system": []}
    for e in inc.report.toolbox.evidence.values() if inc.report.toolbox else []:
        out[e.layer].append(e)
    return out


def make_chart(inc):
    """Accuracy, AUC and mean income. Shows the 'after fix' run when there is one."""
    store = inc.attempts[-1].after if inc.attempts else inc.before
    m = store.metrics()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    for ax, col, title in zip(axes, ["accuracy", "auc", "mean_income"],
                              ["Accuracy", "AUC", "Mean income (log scale)"]):
        ax.plot(m.day, m[col], marker="o", ms=3)
        ax.set_title(title)
        ax.set_xlabel("day")
        if col == "mean_income":
            ax.set_yscale("log")
            ax.set_ylim(10, 1e5)     
        if inc.fault:
            ax.axvline(FAULT_DAY, color="red", ls="--", lw=1)
        if inc.attempts:
            ax.axvline(OBSERVE_DAYS, color="green", ls="--", lw=1)
    axes[0].text(0.02, 0.04, "red = fault injected (demo only)\ngreen = fix applied",
                 transform=axes[0].transAxes, fontsize=7)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    from .sim.simulator import Simulator
    sim = Simulator()
    for label, (fault, variant) in SCENARIOS.items():
        inc = start_incident(sim, fault, variant, 7, RuleReasoner())
        steps = [status(inc)]
        while status(inc) == "awaiting_approval":
            apply_fix(sim, inc)
            steps.append(status(inc))
        top = inc.report.top.cause if inc.report.top else "-"
        fig = make_chart(inc)
        plt.close(fig)
        print(f"{label:<36} found={top:<14} path={' -> '.join(steps)}")

    # a human who declines
    inc = start_incident(sim, "bad_deploy", "unit", 7, RuleReasoner())
    decline(inc)
    print("declined flow:", status(inc))