"""
baseline.py - the "single prompt" competitor.

The baseline LLM gets a text summary of the raw telemetry (daily metrics table,
deploy history, non-routine log patterns) and answers ONCE with a cause.
It has NO tools (no PSI/KS, no model replay, no data-quality checks),
NO validator, and NO recovery check.

The cause definitions below are copied from llm_reasoner.py so both systems
get exactly the same list of candidate causes. Keep them identical.
"""
import re

from ..agent.hypotheses import ACTION_FOR

SYSTEM_PROMPT = """You are an ML reliability investigator. A machine-learning service (a loan default model) started performing badly. You receive a summary of its telemetry.

Decide the root cause. The candidate causes are exactly these three:
- data_drift: the real world changed gradually (for example customers earn less). The inputs are valid but different, and the true outcomes shift too. Typical fix: retrain on recent data.
- schema_change: a data pipeline change broke the inputs (wrong units, missing values) while the real world did not change. Typical fix: revert the pipeline change.
- bad_deploy: a newly deployed model version is worse than the previous one. Typical fix: roll back the model.

Reply with JSON only."""

USER_TEMPLATE = """Daily metrics (CSV, one row per day):
{table}

Deploy history:
{deploys}

Non-routine log patterns (level, source, pattern, first day, last day, count):
{logs}

Reply with JSON in exactly this shape:
{{"cause": "data_drift | schema_change | bad_deploy", "reasoning": "short explanation"}}"""

METRIC_COLS = ["day", "model_version", "pipeline_version", "accuracy", "auc",
               "actual_default_rate", "max_null_rate", "mean_age", "mean_income",
               "mean_credit_score", "mean_loan_amount", "mean_employment_years",
               "mean_late_payments"]


def summarize(store):
    table = store.metrics()[METRIC_COLS].to_csv(index=False)
    deploys = "\n".join(f"day {r.day}: {r.component} {r.version} ({r.note})"
                        for r in store.deploys().itertuples())
    logs = store.logs()
    logs = logs[~((logs.source == "serving") & logs.message.str.startswith("batch"))]
    logs = logs.assign(pattern=logs.message.map(lambda s: re.sub(r"\d+", "#", s)))
    lines = []
    for (level, source, pattern), g in logs.groupby(["level", "source", "pattern"]):
        lines.append(f"{level}, {source}, {pattern}, {g.day.min()}, {g.day.max()}, {len(g)}")
    return USER_TEMPLATE.format(table=table, deploys=deploys, logs="\n".join(lines) or "none")


def diagnose(client, store):
    """Returns (cause or None, error text or '')."""
    try:
        data = client.complete_json(SYSTEM_PROMPT, summarize(store))
        cause = str(data.get("cause", "")).strip()
        if cause not in ACTION_FOR:
            return None, f"invalid cause '{cause}'"
        return cause, ""
    except Exception as err:
        return None, f"{type(err).__name__}: {err}"