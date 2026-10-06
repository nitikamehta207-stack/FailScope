"""
tools.py - the deterministic investigation tools.

Every number here is computed by plain code (no LLM). The LLM will only
read the Evidence these tools return and interpret it.

Two time windows are used everywhere:
  baseline : days assumed healthy (default: the first 10 days)
  recent   : the latest days we want to explain (default: the last 5 days)
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

from ..sim.dataset import FEATURES
from .evidence import Evidence

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

# Which layer does each metric belong to?
METRIC_LAYER = {"accuracy": "model", "auc": "model", "pred_default_rate": "model",
                "actual_default_rate": "model", "max_null_rate": "data",
                "latency_p50_ms": "system", "latency_p95_ms": "system"}
for feat in FEATURES:
    METRIC_LAYER[f"mean_{feat}"] = "data"


def _f(x):
    return round(float(x), 4)


def _fmt(x):
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.3f}"


def _psi(base, recent, bins=10):
    """Population Stability Index: how different two distributions are."""
    edges = np.unique(np.quantile(base, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    b = np.histogram(base, edges)[0] / len(base)
    r = np.histogram(recent, edges)[0] / len(recent)
    b, r = np.clip(b, 1e-4, None), np.clip(r, 1e-4, None)
    return float(np.sum((r - b) * np.log(r / b)))


def _pattern(msg):
    """Turn 'imputed 277 missing values' into 'imputed # missing values'."""
    return re.sub(r"\d+", "#", msg)


class Toolbox:
    def __init__(self, store, baseline=(0, 9), recent=None):
        self.store = store
        self.metrics_df = store.metrics()
        self.req = store.requests()
        self.logs_df = store.logs()
        self.deploys_df = store.deploys()
        n_days = int(self.metrics_df["day"].max()) + 1
        self.baseline = baseline
        self.recent = recent or (n_days - 5, n_days - 1)
        self.models = {"v1": joblib.load(MODELS_DIR / "model_v1.joblib"),
                       "v2": joblib.load(MODELS_DIR / "model_v2.joblib")}
        self.live_version = self.metrics_df["model_version"].iloc[-1]
        self.evidence = {}          # id -> Evidence (the validator will use this later)
        self._counter = 0
        base_rows = self._rows(self.baseline)
        self.medians = base_rows[FEATURES].median()

    # ---------- helpers ----------
    def _new(self, layer, tool, finding, data, anomalous):
        self._counter += 1
        ev = Evidence(f"E{self._counter}", layer, tool, finding, data, bool(anomalous))
        self.evidence[ev.id] = ev
        return ev

    def _rows(self, window):
        return self.req[(self.req.day >= window[0]) & (self.req.day <= window[1])]

    def _mrows(self, window):
        m = self.metrics_df
        return m[(m.day >= window[0]) & (m.day <= window[1])]

    # ---------- tool 1 ----------
    def get_metric_series(self, metric):
        m = self.metrics_df
        values = m[metric].to_numpy(dtype=float)
        base = self._mrows(self.baseline)[metric].to_numpy(dtype=float)
        mu = base.mean()
        sd = max(base.std(), 0.01 * abs(mu), 1e-3)       # floor so flat metrics don't break
        rec = self._mrows(self.recent)[metric].mean()

        change_day = None
        start = self.baseline[1] + 1
        for d in range(start, len(values) - 1):
            a, b = values[d] - mu, values[d + 1] - mu
            if abs(a) > 3 * sd and abs(b) > 3 * sd and a * b > 0:
                change_day = int(m["day"].iloc[d])
                break

        data = {"metric": metric, "baseline_mean": _f(mu), "recent_mean": _f(rec),
                "change_day": change_day}
        if change_day is None:
            finding = f"{metric} stayed stable (baseline {_fmt(mu)}, recent {_fmt(rec)})."
        else:
            shift = rec - mu
            first = values[m.index[m.day == change_day][0]] - mu
            frac = first / shift if shift != 0 else 1.0
            if abs(shift) < 10 * sd:                      # too noisy to judge the shape
                shape = "unclear"
                how = "changed"
            elif frac >= 0.75:
                shape = "abrupt"
                how = "changed abruptly (within one day)"
            else:
                shape = "gradual"
                how = "changed gradually (over several days)"
            data["shape"] = shape
            data["first_day_share_of_total_shift"] = _f(frac)
            pct = f" ({100 * shift / mu:+.1f}%)" if mu else ""
            finding = (f"{metric} {how} from {_fmt(mu)} to {_fmt(rec)}{pct}, "
                       f"starting day {change_day}.")
        return self._new(METRIC_LAYER[metric], "get_metric_series", finding, data,
                         change_day is not None)

    # ---------- tool 2 ----------
    def compare_distributions(self):
        base, rec = self._rows(self.baseline), self._rows(self.recent)
        result = {}
        for f in FEATURES:
            b, r = base[f].dropna().to_numpy(), rec[f].dropna().to_numpy()
            result[f] = {"psi": _f(_psi(b, r)), "ks": _f(ks_2samp(b, r).statistic)}
        flagged = {f: v for f, v in result.items() if v["psi"] > 0.25}
        if flagged:
            parts = ", ".join(f"{f} (PSI={v['psi']:.2f}, KS={v['ks']:.2f})"
                              for f, v in flagged.items())
            finding = f"Significant distribution shift vs baseline in: {parts}."
        else:
            finding = "No feature shows a significant distribution shift (all PSI <= 0.25)."
        return self._new("data", "compare_distributions", finding,
                         {"per_feature": result, "flagged": list(flagged)}, bool(flagged))

    # ---------- tool 3 ----------
    def check_data_quality(self):
        base, rec = self._rows(self.baseline), self._rows(self.recent)
        report, issues = {}, []
        for f in FEATURES:
            b, r = base[f].dropna(), rec[f].dropna()
            null_b, null_r = base[f].isna().mean(), rec[f].isna().mean()
            ratio = r.mean() / b.mean() if b.mean() != 0 else 1.0
            out = float(((r < b.min()) | (r > b.max())).mean()) if len(r) else 0.0
            report[f] = {"null_rate_baseline": _f(null_b), "null_rate_recent": _f(null_r),
                         "mean_ratio": _f(ratio), "out_of_range_share": _f(out)}
            if null_r - null_b > 0.02:
                issues.append(f"{f}: null rate {null_b:.0%} -> {null_r:.0%}")
            if out > 0.2 or ratio < 0.2 or ratio > 5:
                issues.append(f"{f}: values outside normal range "
                              f"({out:.0%} out of range, mean x{ratio:.4g} vs baseline)")
        finding = ("Data quality problems: " + "; ".join(issues) + ".") if issues \
            else "Data quality looks fine (no null spike, values within normal range)."
        return self._new("data", "check_data_quality", finding,
                         {"per_feature": report, "issues": issues}, bool(issues))

    # ---------- tool 4 ----------
    def get_deploy_history(self, around_day=None, window=1):
        d = self.deploys_df
        d = d[d.day > 0]                                  # ignore the initial release
        if around_day is not None:
            d = d[(d.day >= around_day - window) & (d.day <= around_day + window)]
        rows = d.to_dict("records")
        if rows:
            txt = "; ".join(f"day {r['day']}: {r['component']} {r['version']} ({r['note']})"
                            for r in rows)
            where = f" near day {around_day}" if around_day is not None else ""
            finding = f"Deploys{where}: {txt}."
        else:
            where = f" near day {around_day}" if around_day is not None else ""
            finding = f"No deploys{where}."
        return self._new("system", "get_deploy_history", finding,
                         {"deploys": rows, "around_day": around_day}, bool(rows))

    # ---------- tool 5 ----------
    def evaluate_model_version(self, version):
        model = self.models[version]

        def score(rows):
            X = rows[FEATURES].fillna(self.medians)
            proba = model.predict_proba(X)[:, 1]
            return (_f(((proba >= 0.5) == rows["label"]).mean()),
                    _f(roc_auc_score(rows["label"], proba)))

        acc_b, auc_b = score(self._rows(self.baseline))
        acc_r, auc_r = score(self._rows(self.recent))
        live_base_acc = float(self._mrows(self.baseline)["accuracy"].mean())
        live = self.live_version
        weaker = acc_b < live_base_acc - 0.03             # bad even on healthy traffic
        degraded = acc_r < acc_b - 0.03                   # fine before, bad now
        data = {"version": version, "is_live": version == live,
                "baseline": {"accuracy": acc_b, "auc": auc_b},
                "recent": {"accuracy": acc_r, "auc": auc_r},
                "weaker_than_original_on_healthy_data": weaker,
                "degraded_on_recent_data": degraded}
        finding = (f"Model {version}{' (live)' if version == live else ''} replayed: "
                   f"baseline acc={acc_b:.3f}/auc={auc_b:.3f}, "
                   f"recent acc={acc_r:.3f}/auc={auc_r:.3f}")
        if weaker:
            finding += " -> weak even on healthy baseline traffic."
        elif degraded:
            finding += " -> good on baseline traffic but degraded on recent traffic."
        else:
            finding += " -> performs normally on recent traffic."
        return self._new("model", "evaluate_model_version", finding, data, weaker or degraded)

    # ---------- tool 6 ----------
    def read_logs(self, day_from=None, day_to=None):
        day_from = self.recent[0] if day_from is None else day_from
        day_to = self.recent[1] if day_to is None else day_to
        L = self.logs_df
        base = L[(L.day >= self.baseline[0]) & (L.day <= self.baseline[1])]
        win = L[(L.day >= day_from) & (L.day <= day_to)]
        seen = set(base["message"].map(_pattern))
        new = win[~win["message"].map(_pattern).isin(seen)]
        new_patterns = (new.assign(p=new["message"].map(_pattern))
                        .groupby(["level", "source", "p"]).size().reset_index(name="count"))
        items = [{"level": r.level, "source": r.source, "message": r.p, "count": int(r["count"])}
                 for _, r in new_patterns.iterrows()]
        if items:
            txt = "; ".join(f"{i['level']} {i['source']}: '{i['message']}' x{i['count']}"
                            for i in items)
            finding = f"New log patterns on days {day_from}-{day_to} (not seen in baseline): {txt}."
        else:
            finding = f"No new log patterns on days {day_from}-{day_to} compared to baseline."
        return self._new("system", "read_logs", finding,
                         {"day_from": day_from, "day_to": day_to, "new_patterns": items},
                         bool(items))