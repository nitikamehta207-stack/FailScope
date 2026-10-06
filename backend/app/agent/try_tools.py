"""
try_tools.py - runs every tool on every scenario and prints the evidence.
Run from the FailScope folder:  python -m backend.app.agent.try_tools
"""
from ..sim.simulator import Simulator
from .tools import Toolbox

SCENARIOS = [("healthy", None, "unit"), ("data_drift", "data_drift", "unit"),
             ("schema_change (unit)", "schema_change", "unit"),
             ("schema_change (nulls)", "schema_change", "nulls"),
             ("bad_deploy", "bad_deploy", "unit")]


def investigate(store):
    tb = Toolbox(store)
    acc = tb.get_metric_series("accuracy")
    day = acc.data["change_day"]
    for m in ["auc", "actual_default_rate", "mean_income", "max_null_rate"]:
        tb.get_metric_series(m)
    tb.compare_distributions()
    tb.check_data_quality()
    tb.get_deploy_history(around_day=day)
    for version in dict.fromkeys([tb.live_version, "v1"]):   # live model, plus v1 as the old reference
        tb.evaluate_model_version(version)
    tb.read_logs(day_from=(day - 1) if day else None, day_to=(day + 1) if day else None)
    return tb


if __name__ == "__main__":
    sim = Simulator()
    for name, fault, variant in SCENARIOS:
        print("=" * 100)
        print("SCENARIO:", name)
        tb = investigate(sim.run(fault=fault, variant=variant, seed=0))
        for ev in tb.evidence.values():
            print(ev.short())