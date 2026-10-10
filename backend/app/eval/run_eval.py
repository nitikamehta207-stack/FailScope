"""
run_eval.py - compares three systems on fresh seeds the agent was never tuned on.

  rules    : FailScope with the rule-based reasoner (no LLM calls)
  llm      : FailScope with the LLM reasoner (tools + validator + verify + retry)
  baseline : one LLM prompt over a telemetry summary (no tools, no validator, no verify)

Metrics:
  RCA top-1        : was the first diagnosis the true fault?
  fixed 1st try    : did the first fix recover the system (checked by verify_recovery)?
  recovered final  : recovered after verify + up to 2 attempts (baseline has no retry)
  escalated        : the system gave up and asked a human

Run from the FailScope folder:
  python -m backend.app.eval.run_eval --systems rules
  python -m backend.app.eval.run_eval --systems rules,llm,baseline --seeds 10
  python -m backend.app.eval.run_eval --mock --seeds 2      # tests the plumbing only
Results are saved after every run, so you can stop and resume safely.
"""
import argparse
import json
import math
import time
from pathlib import Path

from ..sim.simulator import Simulator
from ..agent.agent import Investigator
from ..agent.hypotheses import ACTION_FOR
from ..agent.llm_reasoner import LLMReasoner
from ..agent.reasoner import RuleReasoner
from ..agent.remediate import run_incident, verify_recovery
from . import baseline

FAULTS = ["data_drift", "schema_change", "bad_deploy"]
EVAL_DIR = Path(__file__).resolve().parents[3] / "eval_results"
OBSERVE_DAYS, TOTAL_DAYS = 23, 30


class NoFallback:
    """In the eval an LLM failure must show up as a failure, not be hidden by the rules."""
    def propose(self, evidence):
        return []


class ThrottledClient:
    """Waits between calls so we stay under the provider's requests-per-minute limit."""
    def __init__(self, client, min_interval=2.5):
        self.client, self.min_interval, self._last = client, min_interval, 0.0

    def complete_json(self, system, user):
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        try:
            return self.client.complete_json(system, user)
        finally:
            self._last = time.time()


class MockClient:
    """Makes up answers. ONLY for testing that the harness runs. Never report its numbers."""
    def complete_json(self, system, user):
        if "Evidence:" in user:
            import re
            ids = re.findall(r"\[(E\d+)\] layer=\w+ tool=\w+ anomalous=yes", user)[:4]
            return {"hypotheses": [
                {"cause": "data_drift", "confidence": 0.9, "evidence_ids": ids, "reasoning": "mock"},
                {"cause": "schema_change", "confidence": 0.6, "evidence_ids": ids, "reasoning": "mock"},
                {"cause": "bad_deploy", "confidence": 0.5, "evidence_ids": ids, "reasoning": "mock"}]}
        return {"cause": "data_drift", "reasoning": "mock"}


def is_rate_limit(err):
    return "429" in err or "rate" in err.lower()


def run_agent_system(sim, investigator, reasoner, fault, seed, variant):
    result = run_incident(sim, fault, seed=seed, variant=variant, investigator=investigator,
                          observe_days=OBSERVE_DAYS, total_days=TOTAL_DAYS)
    top = result.report.top
    failure = ""
    if reasoner is not None and result.report.incident and reasoner.last_source != "llm":
        failure = reasoner.last_error or reasoner.last_source
    return {"guess": top.cause if top else "none",
            "attempts": len(result.attempts),
            "fixed_first_try": bool(result.attempts and result.attempts[0].verification.recovered),
            "recovered_final": bool(result.resolved),
            "escalated": not result.resolved,
            "llm_failure": failure}


def run_baseline(sim, client, fault, seed, variant):
    before = sim.run(n_days=OBSERVE_DAYS, fault=fault, seed=seed, variant=variant)
    cause, err = baseline.diagnose(client, before)
    rec = {"guess": cause or "none", "attempts": 0, "fixed_first_try": False,
           "recovered_final": False, "escalated": False, "llm_failure": err}
    if cause:
        after = sim.run(n_days=TOTAL_DAYS, fault=fault, seed=seed, variant=variant,
                        fix=ACTION_FOR[cause], fix_day=OBSERVE_DAYS)
        ok = bool(verify_recovery(after, OBSERVE_DAYS).recovered)
        rec.update(attempts=1, fixed_first_try=ok, recovered_final=ok)
    return rec


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def pct(k, n):
    lo, hi = wilson(k, n)
    return f"{k:>2}/{n} = {100 * k / n:3.0f}%  (95% CI {100 * lo:.0f}-{100 * hi:.0f}%)" if n else "n/a"


def summarize(rows):
    print("\n" + "=" * 100)
    print("RESULTS (one row per run; CI = 95% Wilson interval, wide because n is small)")
    for system in ["rules", "llm", "baseline"]:
        r = [x for x in rows if x["system"] == system]
        if not r:
            continue
        n = len(r)
        correct = sum(x["guess"] == x["fault"] for x in r)
        print(f"\n[{system}]  runs = {n}")
        print(f"  RCA top-1        : {pct(correct, n)}")
        print(f"  fixed on 1st try : {pct(sum(x['fixed_first_try'] for x in r), n)}")
        print(f"  recovered final  : {pct(sum(x['recovered_final'] for x in r), n)}")
        print(f"  escalated to human: {sum(x['escalated'] for x in r)}    "
              f"LLM failures: {sum(bool(x['llm_failure']) for x in r)}")
        for f in FAULTS:
            rf = [x for x in r if x["fault"] == f]
            if rf:
                print(f"    {f:<14} correct {sum(x['guess'] == f for x in rf)}/{len(rf)}   "
                      f"recovered {sum(x['recovered_final'] for x in rf)}/{len(rf)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="rules,llm,baseline")
    ap.add_argument("--seeds", type=int, default=10, help="seeds per fault")
    ap.add_argument("--start-seed", type=int, default=100, help="fresh seeds, never used while building")
    ap.add_argument("--mock", action="store_true", help="fake LLM, only tests that the code runs")
    args = ap.parse_args()
    systems = [s.strip() for s in args.systems.split(",")]

    EVAL_DIR.mkdir(exist_ok=True)
    path = EVAL_DIR / ("results_mock.jsonl" if args.mock else "results.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    done = {(r["system"], r["fault"], r["seed"]) for r in rows}

    sim = Simulator()
    client = None
    if any(s in systems for s in ("llm", "baseline")):
        if args.mock:
            client = ThrottledClient(MockClient(), min_interval=0)
        else:
            from ..agent.llm_client import LLMClient
            client = ThrottledClient(LLMClient(), min_interval=6)
    llm_reasoner = LLMReasoner(client=client, fallback=NoFallback()) if client else None
    investigators = {"rules": (Investigator(RuleReasoner()), None),
                     "llm": (Investigator(llm_reasoner), llm_reasoner) if llm_reasoner else None}

    seeds = range(args.start_seed, args.start_seed + args.seeds)
    for fault in FAULTS:
        for seed in seeds:
            variant = "unit" if seed % 2 == 0 else "nulls"
            for system in systems:
                if (system, fault, seed) in done:
                    continue
                if system == "baseline":
                    rec = run_baseline(sim, client, fault, seed, variant)
                else:
                    inv, reasoner = investigators[system]
                    rec = run_agent_system(sim, inv, reasoner, fault, seed, variant)
                if rec["llm_failure"] and is_rate_limit(rec["llm_failure"]):
                    print(f"  rate limit on {system} {fault} seed={seed}. Not saved, run again later.")
                    continue
                rec.update(system=system, fault=fault, seed=seed, variant=variant)
                rows.append(rec)
                with path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                flag = "OK " if rec["guess"] == fault else "BAD"
                print(f"  {system:<8} {fault:<14} seed={seed} -> {rec['guess']:<14} {flag} "
                      f"recovered={'Y' if rec['recovered_final'] else 'N'}"
                      + (f"   [LLM issue: {rec['llm_failure'][:60]}]" if rec["llm_failure"] else ""))

    summarize(rows)

    if "rules" in systems:
        fa = sum(Investigator(RuleReasoner()).investigate(
                 sim.run(n_days=OBSERVE_DAYS, fault=None, seed=s)).incident for s in seeds)
        print(f"\nFalse alarms on {len(seeds)} healthy runs (same triage for rules and llm): {fa}/{len(seeds)}")
    if args.mock:
        print("\nMOCK MODE: these numbers are meaningless, they only prove the harness runs.")


if __name__ == "__main__":
    main()