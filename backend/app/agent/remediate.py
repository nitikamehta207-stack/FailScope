"""
remediate.py - apply the recommended fix, then CHECK whether the system recovered.

Flow for each validated hypothesis (best first, max 2 attempts):
    human approval -> apply fix in sandbox -> verify_recovery -> done, or try next hypothesis
"""
from dataclasses import dataclass, field

from .evidence import Evidence

ACC_TOLERANCE = 0.03      # recovered accuracy may be at most 3 points below the healthy baseline
NULL_TOLERANCE = 0.02     # null rate may be at most 2 points above the healthy baseline
MIN_GAP_CLOSED = 0.4      # OR: the fix must win back at least 40% of the accuracy that was lost.
                          # (After real data drift the new world can be harder than the old one,
                          #  so "back to the old baseline" is not always possible.)


def verify_recovery(store_after, fix_day, baseline=(0, 9), pre_days=5, n=1):
    """Compare the days after the fix against the healthy baseline (deterministic, no LLM)."""
    m = store_after.metrics()
    base = m[(m.day >= baseline[0]) & (m.day <= baseline[1])]
    pre = m[(m.day >= fix_day - pre_days) & (m.day < fix_day)]
    post = m[m.day >= fix_day]

    base_acc, pre_acc, post_acc = (float(x.accuracy.mean()) for x in (base, pre, post))
    base_null, post_null = float(base.max_null_rate.mean()), float(post.max_null_rate.mean())
    lost = base_acc - pre_acc
    gap_closed = (post_acc - pre_acc) / lost if lost > 0.01 else 1.0
    acc_ok = post_acc >= base_acc - ACC_TOLERANCE or gap_closed >= MIN_GAP_CLOSED
    null_ok = post_null <= base_null + NULL_TOLERANCE
    recovered = acc_ok and null_ok

    if recovered:
        finding = (f"Recovered: accuracy {pre_acc:.3f} (before fix) -> {post_acc:.3f} (after fix), "
                   f"healthy baseline {base_acc:.3f} ({gap_closed:.0%} of the lost accuracy won back).")
    else:
        why = []
        if not acc_ok:
            why.append(f"accuracy is {post_acc:.3f} (baseline {base_acc:.3f}), only {gap_closed:.0%} of the lost accuracy won back")
        if not null_ok:
            why.append(f"null rate {post_null:.2f} is still high")
        finding = "NOT recovered: " + "; ".join(why) + "."
    return Evidence(f"V{n}", "model", "verify_recovery", finding,
                    {"baseline_acc": round(base_acc, 4), "pre_fix_acc": round(pre_acc, 4),
                     "post_fix_acc": round(post_acc, 4), "gap_closed": round(gap_closed, 3), "baseline_null": round(base_null, 4),
                     "post_null": round(post_null, 4), "recovered": recovered},
                    anomalous=not recovered)


@dataclass
class Attempt:
    hypothesis: object
    approved: bool
    verification: object = None       # Evidence from verify_recovery (None if not approved)

    @property
    def recovered(self):
        return bool(self.verification and self.verification.data["recovered"])


@dataclass
class Outcome:
    attempts: list = field(default_factory=list)
    recovered: bool = False
    fixed_by: object = None           # the hypothesis whose fix worked
    trace: list = field(default_factory=list)


class Remediator:
    def __init__(self, max_attempts=2, approve=None):
        self.max_attempts = max_attempts
        # Human-in-the-loop. Default = auto-approve (CLI/eval). The UI will pass a real button.
        self.approve = approve or (lambda hypothesis: True)

    def run(self, report, sandbox):
        out = Outcome()
        if not report.incident:
            out.trace.append("No incident, nothing to fix.")
            return out
        for i, h in enumerate(report.valid[:self.max_attempts], start=1):
            if not self.approve(h):
                out.attempts.append(Attempt(h, approved=False))
                out.trace.append(f"Attempt {i}: human did not approve '{h.action}'. Escalating.")
                break
            out.trace.append(f"Attempt {i}: applying '{h.action}' (for cause '{h.cause}')...")
            after = sandbox.apply_fix(h.action)
            v = verify_recovery(after, sandbox.fix_day, n=i)
            out.attempts.append(Attempt(h, True, v))
            out.trace.append(f"Attempt {i}: {v.finding}")
            if v.data["recovered"]:
                out.recovered, out.fixed_by = True, h
                break
        if not out.recovered and out.attempts and all(a.approved for a in out.attempts):
            out.trace.append("Fix did not recover the system. Escalate to a human.")
        return out