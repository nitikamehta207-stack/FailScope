"""
simulator.py - runs the fake ML service day by day and records telemetry.

Each "day" = one batch of loan applications. The service:
  1. receives raw applications (maybe affected by a fault)
  2. fills missing values, predicts with the current model version
  3. later, true labels arrive, so we can measure accuracy
  4. everything is written to the TelemetryStore

Faults (only parameters change, no new code paths):
  "data_drift"    : the real world changes slowly (income falls), no deploy
  "schema_change" : a pipeline update changes a field (income unit, or nulls)
  "bad_deploy"    : model v2 is rolled out, and it is worse than v1
"""
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .dataset import generate_data, FEATURES
from .telemetry import TelemetryStore

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
FAULTS = ("data_drift", "schema_change", "bad_deploy")
FIXES = ("rollback_model", "revert_pipeline_change", "retrain_on_recent")


class Simulator:
    def __init__(self):
        self.models = {"v1": joblib.load(MODELS_DIR / "model_v1.joblib"),
                       "v2": joblib.load(MODELS_DIR / "model_v2.joblib")}
        # Healthy reference data: used for imputing and (later) for drift checks
        self.reference = generate_data(5000, seed=2)
        self.medians = self.reference[FEATURES].median()

    def _retrain_on_recent(self, store, day_from, day_to):
        """Fix: train a fresh model (v3) on the latest labelled traffic."""
        req = store.requests()
        recent = req[(req.day >= day_from) & (req.day <= day_to)]
        X = recent[FEATURES].fillna(self.medians)
        model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=0)
        model.fit(X, recent["label"])
        self.models["v3"] = model
        return "v3"

    def run(self, n_days=30, batch_size=1000, fault=None, fault_day=15,
            seed=0, variant="unit", ramp_days=4, fix=None, fix_day=None):
        assert fault is None or fault in FAULTS
        assert fix is None or fix in FIXES
        self.models.pop("v3", None)          # a retrained model never leaks between runs
        pipeline_reverted = False
        store = TelemetryStore()
        store.ground_truth = {"fault": fault, "fault_day": fault_day if fault else None,
                              "variant": variant if fault == "schema_change" else None,
                              "seed": seed}
        rng = np.random.default_rng(1000 + seed)
        model_version, pipeline_version = "v1", "p1"
        store.add_deploy(0, "model", "v1", "initial release")
        store.add_deploy(0, "pipeline", "p1", "initial release")
        store.add_log(0, "INFO", "deploy", "model v1 and pipeline p1 are live")

        for day in range(n_days):
            # --- deploys that happen on fault day ---
            if day == fault_day and fault == "bad_deploy":
                model_version = "v2"
                store.add_deploy(day, "model", "v2", "scheduled model refresh")
                store.add_log(day, "INFO", "deploy", "model v2 rolled out")
            if day == fault_day and fault == "schema_change":
                pipeline_version = "p2"
                store.add_deploy(day, "pipeline", "p2", "preprocessing update")
                store.add_log(day, "INFO", "deploy", "pipeline p2 rolled out")

            # --- remediation: a fix applied on fix_day (used after the agent's diagnosis) ---
            if fix and day == fix_day:
                if fix == "rollback_model":
                    model_version = "v1"
                    store.add_deploy(day, "model", "v1", "rollback to v1 (remediation)")
                    store.add_log(day, "INFO", "deploy", "model rolled back to v1")
                elif fix == "revert_pipeline_change":
                    pipeline_version, pipeline_reverted = "p1", True
                    store.add_deploy(day, "pipeline", "p1", "revert preprocessing update (remediation)")
                    store.add_log(day, "INFO", "deploy", "pipeline reverted to p1")
                elif fix == "retrain_on_recent":
                    model_version = self._retrain_on_recent(store, day - 5, day - 1)
                    store.add_deploy(day, "model", model_version, "retrained on last 5 days (remediation)")
                    store.add_log(day, "INFO", "deploy", "model v3 retrained and rolled out")

            # --- parameters for today's data ---
            drift, corrupt = None, None
            if fault == "data_drift" and day >= fault_day:
                progress = min(1.0, (day - fault_day + 1) / ramp_days)   # gradual change
                drift = {"income_scale": 1 + (0.6 - 1) * progress}
            if fault == "schema_change" and day >= fault_day and not pipeline_reverted:
                corrupt = {"income_scale": 0.001} if variant == "unit" else {"credit_null_rate": 0.3}

            # --- generate and serve the batch ---
            df = generate_data(batch_size, seed=10_000 + seed * 100 + day,
                               drift=drift, corrupt=corrupt)
            X = df[FEATURES]
            null_rates = X.isna().mean()
            X_served = X.fillna(self.medians)
            model = self.models[model_version]
            proba = model.predict_proba(X_served)[:, 1]
            pred = (proba >= 0.5).astype(int)
            label = df["default"].to_numpy()

            # --- request-level records ---
            req = X.copy()
            req.insert(0, "day", day)
            req["model_version"] = model_version
            req["proba"] = proba.round(4)
            req["pred"] = pred
            req["label"] = label
            store.add_requests(req)

            # --- latency (random, healthy in every scenario) ---
            latency = rng.lognormal(np.log(40), 0.25, batch_size)

            # --- daily metrics ---
            row = {"day": day, "model_version": model_version,
                   "pipeline_version": pipeline_version, "n_requests": batch_size,
                   "accuracy": round(float((pred == label).mean()), 4),
                   "auc": round(float(roc_auc_score(label, proba)), 4),
                   "pred_default_rate": round(float(pred.mean()), 4),
                   "actual_default_rate": round(float(label.mean()), 4),
                   "max_null_rate": round(float(null_rates.max()), 4),
                   "latency_p50_ms": round(float(np.percentile(latency, 50)), 1),
                   "latency_p95_ms": round(float(np.percentile(latency, 95)), 1)}
            for f in FEATURES:
                row[f"mean_{f}"] = round(float(X[f].mean()), 2)
            store.add_metrics(row)

            # --- logs ---
            store.add_log(day, "INFO", "serving",
                          f"batch {day} processed n={batch_size} model={model_version}")
            n_imputed = int(X.isna().sum().sum())
            if n_imputed > 0:
                store.add_log(day, "WARN", "preprocess", f"imputed {n_imputed} missing values")
            if rng.random() < 0.08:                      # harmless noise in every scenario
                store.add_log(day, "WARN", "serving", "slow request detected (retry succeeded)")
            if rng.random() < 0.05:
                store.add_log(day, "WARN", "feature_store", "cache miss rate above usual")

        return store


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sim = Simulator()
    scenarios = [("healthy", None, "unit"), ("data_drift", "data_drift", "unit"),
                 ("schema_change (unit)", "schema_change", "unit"),
                 ("schema_change (nulls)", "schema_change", "nulls"),
                 ("bad_deploy", "bad_deploy", "unit")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for name, fault, variant in scenarios:
        store = sim.run(fault=fault, variant=variant, seed=0)
        m = store.metrics()
        before = m[m.day < 15]
        after = m[m.day >= 18]
        print(f"{name:<24} acc before={before.accuracy.mean():.3f} after={after.accuracy.mean():.3f} | "
              f"auc after={after.auc.mean():.3f} | null after={after.max_null_rate.mean():.2f} | "
              f"mean_income after={after.mean_income.mean():.0f}")
        axes[0].plot(m.day, m.accuracy, label=name)
        axes[1].plot(m.day, m.auc, label=name)
        axes[2].plot(m.day, m.mean_income, label=name)
        store.save(Path(__file__).resolve().parents[3] / "telemetry_runs" / name.replace(" ", "_"))
    for ax, t in zip(axes, ["accuracy", "auc", "mean_income"]):
        ax.set_title(t); ax.set_xlabel("day"); ax.axvline(15, color="gray", ls="--")
    axes[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(Path(__file__).resolve().parents[3] / "telemetry_runs" / "overview.png", dpi=110)
    print("saved plots and telemetry in telemetry_runs/")