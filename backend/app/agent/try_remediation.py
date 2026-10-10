"""
try_remediation.py - full loop: investigate -> fix -> verify recovery.
Run from the FailScope folder:  python -m backend.app.agent.try_remediation
"""
from ..sim.simulator import Simulator
from .agent import Investigator
from .remediate import run_incident, format_incident

FAULTS = ["data_drift", "schema_change", "bad_deploy"]


def main():
    sim = Simulator()

    print("=" * 90)
    print("PART 1: one full incident per fault (seed 0)")
    for fault in FAULTS:
        print("-" * 90)
        print("FAULT (hidden from agent):", fault)
        print(format_incident(run_incident(sim, fault, seed=0)))

    print("=" * 90)
    print("PART 2: 12 runs (schema_change alternates unit / nulls)")
    resolved, total = 0, 0
    for fault in FAULTS:
        for seed in range(4):
            variant = "unit" if seed % 2 == 0 else "nulls"
            r = run_incident(sim, fault, seed=seed, variant=variant)
            guess = r.report.top.cause if r.report.top else "none"
            resolved += r.resolved
            total += 1
            print(f"  {fault:<14} seed={seed} -> found {guess:<14} "
                  f"attempts={len(r.attempts)} recovered={'YES' if r.resolved else 'NO'}")
    print(f"Recovered: {resolved}/{total}")

    print("=" * 90)
    print("PART 3: what if the agent's FIRST guess is wrong? (schema_change fault)")
    agent = Investigator()
    base = run_incident(sim, "schema_change", seed=0)
    wrong = next(x.hypothesis for x in base.report.rejected if x.hypothesis.cause == "data_drift")
    right = base.report.top
    print(format_incident(run_incident(sim, "schema_change", seed=0,
                                       investigator=agent, hypotheses=[wrong, right])))


if __name__ == "__main__":
    main()