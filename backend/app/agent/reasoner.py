"""
reasoner.py - reads Evidence and proposes hypotheses.

RuleReasoner is a simple stand-in that needs no API key. Later an LLM-based
reasoner will replace it. Both have the same method:

    propose(evidence: dict[str, Evidence]) -> list[Hypothesis]
"""
from .hypotheses import Hypothesis, ACTION_FOR


class _Index:
    """Small helper to look up evidence by tool name."""
    def __init__(self, evidence):
        self.items = list(evidence.values())

    def all(self, tool):
        return [e for e in self.items if e.tool == tool]

    def first(self, tool):
        found = self.all(tool)
        return found[0] if found else None


class RuleReasoner:
    def propose(self, evidence):
        idx = _Index(evidence)

        # ---- pull out the clues once ----
        dep = idx.first("get_deploy_history")
        rows = dep.data["deploys"] if dep else []
        model_deploy = dep if any(r["component"] == "model" for r in rows) else None
        pipe_deploy = dep if any(r["component"] == "pipeline" for r in rows) else None
        no_deploy = dep if (dep and not rows) else None

        dist = idx.first("compare_distributions")
        dist_shift = dist if (dist and dist.anomalous) else None
        dist_ok = dist if (dist and not dist.anomalous) else None
        dq = idx.first("check_data_quality")
        dq_bad = dq if (dq and dq.anomalous) else None
        dq_ok = dq if (dq and not dq.anomalous) else None

        evals = idx.all("evaluate_model_version")
        live = next((e for e in evals if e.data["is_live"]), None)
        others = [e for e in evals if not e.data["is_live"]]
        live_weak = live if (live and live.data["weaker_than_original_on_healthy_data"]) else None
        live_degraded = live if (live and live.data["degraded_on_recent_data"]
                                 and not live.data["weaker_than_original_on_healthy_data"]) else None
        old_fine = next((e for e in others if not e.anomalous), None)

        series = idx.all("get_metric_series")
        feat = [e for e in series if e.anomalous and e.data["metric"].startswith("mean_")]
        nulls = [e for e in series if e.anomalous and e.data["metric"] == "max_null_rate"]
        gradual = next((e for e in feat if e.data.get("shape") == "gradual"), None)
        abrupt = next((e for e in feat + nulls if e.data.get("shape") == "abrupt"), None)
        rate = next((e for e in series if e.data["metric"] == "actual_default_rate"), None)
        rate_changed = rate if (rate and rate.anomalous) else None
        rate_stable = rate if (rate and not rate.anomalous) else None
        logs = idx.first("read_logs")
        prep_warn = logs if (logs and any(p["level"] in ("WARN", "ERROR") and p["source"] == "preprocess"
                                          for p in logs.data["new_patterns"])) else None

        # ---- each cause = list of (weight, description, evidence found or None) ----
        rules = {
            "bad_deploy": [
                (3, "a model deploy happened right when the problem started", model_deploy),
                (3, "the live model is weak even on healthy baseline traffic", live_weak),
                (3, "the previous model still performs normally on recent traffic", old_fine),
                (1, "inputs look healthy (no distribution shift, no data-quality issue)",
                 [dist_ok, dq_ok]),
            ],
            "data_drift": [
                (2, "feature distributions shifted significantly", dist_shift),
                (1, "data quality is fine (no nulls or broken values)", dq_ok),
                (2, "the shift happened gradually, not in one day", gradual),
                (3, "the real default rate changed too (the world itself changed)", rate_changed),
                (1, "no deploy happened around the change", no_deploy),
                (1, "the model was fine before but degraded on recent traffic", live_degraded),
            ],
            "schema_change": [
                (3, "a pipeline change was deployed right when the problem started", pipe_deploy),
                (3, "data quality broke (null spike or values out of normal range)", dq_bad),
                (2, "a data signal changed abruptly within one day", abrupt),
                (2, "the real default rate stayed stable (the world did not change)", rate_stable),
                (1, "preprocessing logged new warnings", prep_warn),
            ],
        }

        hypotheses = []
        for cause, signals in rules.items():
            total = sum(w for w, _, _ in signals)
            got, ids, said, missing = 0, [], [], []
            for w, text, ev in signals:
                evs = [e for e in (ev if isinstance(ev, list) else [ev]) if e is not None]
                satisfied = bool(evs) and (not isinstance(ev, list) or len(evs) == len(ev))
                if satisfied:
                    got += w
                    for e in evs:
                        if e.id not in ids:
                            ids.append(e.id)
                    said.append(f"{text} ({', '.join(e.id for e in evs)})")
                else:
                    missing.append(text)
            reasoning = "Found: " + "; ".join(said) if said else "No supporting clues found."
            if missing:
                reasoning += ". Not found: " + "; ".join(missing)
            hypotheses.append(Hypothesis(cause, round(got / total, 2), ids, reasoning,
                                         ACTION_FOR[cause]))
        hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        return hypotheses