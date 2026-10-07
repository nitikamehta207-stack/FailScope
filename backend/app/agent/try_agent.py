"""
try_agent.py - runs the investigator on every fault with several seeds.
Run from the FailScope folder:  python -m backend.app.agent.try_agent
"""
from ..sim.simulator import Simulator
from .agent import Investigator, format_report

FAULTS = ["data_drift", "schema_change", "bad_deploy"]


def main():
    sim, agent = Simulator(), Investigator()

    # Show one full report per fault so you can read the agent's reasoning
    for fault in FAULTS:
        store = sim.run(fault=fault, seed=0)
        print("=" * 90)
        print("FAULT (hidden from agent):", fault)
        print(format_report(agent.investigate(store)))

    # Quick check over 4 seeds. schema_change alternates between unit and null variants.
    print("=" * 90)
    print("SEED CHECK")
    correct = 0
    total = 0
    for fault in FAULTS:
        for seed in range(4):
            variant = "unit" if seed % 2 == 0 else "nulls"
            store = sim.run(fault=fault, seed=seed, variant=variant)
            report = agent.investigate(store)
            truth = store.ground_truth["fault"]    # only used here to score the agent
            guess = report.top.cause if report.top else "none"
            ok = guess == truth
            correct += ok
            total += 1
            tag = f"{variant}" if fault == "schema_change" else ""
            print(f"  {fault:<14} seed={seed} {tag:<5} -> {guess:<14} {'OK' if ok else 'WRONG'}")
    print(f"Top-1 correct: {correct}/{total}")

    false_alarms = 0
    for seed in range(4):
        report = agent.investigate(sim.run(fault=None, seed=seed))
        false_alarms += report.incident
    print(f"Healthy runs with a false alarm: {false_alarms}/4")


if __name__ == "__main__":
    main()