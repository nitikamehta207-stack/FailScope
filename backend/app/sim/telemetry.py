"""
telemetry.py - the "black box recorder" of the simulated ML service.

It stores everything the agent is allowed to look at:
  metrics  : one row per day (accuracy, null rate, latency, ...)
  requests : one row per prediction request (features, prediction, label)
  logs     : text log lines
  deploys  : deploy history (model / pipeline versions)

ground_truth (which fault was injected) is saved in a SEPARATE file.
The agent must never read it; only the evaluation code will.
"""
import json
from pathlib import Path
import pandas as pd


class TelemetryStore:
    def __init__(self):
        self._metrics = []
        self._requests = []
        self._logs = []
        self._deploys = []
        self.ground_truth = {}

    # ---- writing ----
    def add_metrics(self, row):
        self._metrics.append(row)

    def add_requests(self, df):
        self._requests.append(df)

    def add_log(self, day, level, source, message):
        self._logs.append({"day": day, "level": level, "source": source, "message": message})

    def add_deploy(self, day, component, version, note):
        self._deploys.append({"day": day, "component": component, "version": version, "note": note})

    # ---- reading (as DataFrames) ----
    def metrics(self):
        return pd.DataFrame(self._metrics)

    def requests(self):
        return pd.concat(self._requests, ignore_index=True)

    def logs(self):
        return pd.DataFrame(self._logs)

    def deploys(self):
        return pd.DataFrame(self._deploys)

    # ---- save / load ----
    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.metrics().to_csv(folder / "metrics.csv", index=False)
        self.requests().to_csv(folder / "requests.csv", index=False)
        self.logs().to_csv(folder / "logs.csv", index=False)
        self.deploys().to_csv(folder / "deploys.csv", index=False)
        (folder / "ground_truth.json").write_text(json.dumps(self.ground_truth, indent=2))