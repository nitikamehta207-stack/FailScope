"""
try_remediate.py - full loop: investigate -> approve -> fix -> verify.
Run from the FailScope folder:  python -m backend.app.agent.try_remediate
"""
import warnings
warnings.filterwarnings("ignore")

from ..sim.simulator import Simulator, FIXES
from ..sim.sandbox import Sandbox
from .agent import Investigator, format_report
from .remediate import Remediator, verify_recovery

FAULTS = ["data_drift", "schema_change", "bad_deploy"]


def main():
    sim, agent, fixer = Simulator(), Investigator(), Remediator()

    # 1) One full story per fault
    for fault in FAULTS:
        sb = Sandbox(sim, fault=fault, seed=0)
        report = agent.investigate(sb.store)
        outcome = fixer.run(report, sb)
        print("=" * 90)
        print("FAULT (hidden from agent):", fault)
        print(format_report(report))
        for line in outcome.trace:
            print("  >>", line)

    # 2) Full loop over 4 seeds
    print("=" * 90)
    print("FULL LOOP CHECK (cause correct + system recovered)")
    n = ok_cause = ok_rec = 0
    for fault in FAULTS:
        for seed in range(4):
            variant = "unit" if seed % 2 == 0 else "nulls"
            sb = Sandbox(sim, fault=fault, seed=seed, variant=variant)
            report = agent.investigate(sb.store)
            outcome = fixer.run(report, sb)
            cause = report.top.cause if report.top else "none"
            n += 1; ok_cause += cause == sb.truth["fault"]; ok_rec += outcome.recovered
            print(f"  {fault:<14} seed={seed} -> cause={cause:<14} attempts={len(outcome.attempts)} "
                  f"recovered={outcome.recovered}")
    print(f"Root cause correct: {ok_cause}/{n} | Recovered: {ok_rec}/{n}")

    # 3) Does verify_recovery catch WRONG fixes? (fault x fix matrix, seed 0)
    print("=" * 90)
    print("WRONG-FIX TEST: recovered? (rows = real fault, columns = fix applied)")
    print(f"  {'':<14}" + "".join(f"{f:<26}" for f in FIXES))
    for fault in FAULTS:
        sb = Sandbox(sim, fault=fault, seed=0)
        cells = []
        for fix in FIXES:
            v = verify_recovery(sb.apply_fix(fix), sb.fix_day)
            cells.append(f"{str(v.data['recovered']):<6}(acc {v.data['post_fix_acc']:.3f})".ljust(26))
        print(f"  {fault:<14}" + "".join(cells))


if __name__ == "__main__":
    main()