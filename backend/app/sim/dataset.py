"""
dataset.py - generates synthetic loan-default data.

Two kinds of problems can be injected:

1) drift   = REAL data drift. The world itself changed (people earn less).
             Features change, and the label is generated from the NEW features.
2) corrupt = Pipeline/schema bug. The world is the same: the label is created
             from the correct features, but the model sees WRONG feature values
             (e.g. income arrives in thousands instead of rupees).
"""
import numpy as np
import pandas as pd

FEATURES = ["age", "income", "credit_score", "loan_amount",
            "employment_years", "late_payments"]


def _make_features(n, rng, drift):
    drift = drift or {}
    age = rng.integers(21, 66, n)
    income = rng.lognormal(mean=10.8, sigma=0.5, size=n)    # median around 50k
    credit = np.clip(rng.normal(650, 80, n), 300, 850)
    loan = income * rng.uniform(0.2, 2.0, n)                # loan size depends on income
    emp = np.clip(age - 21 - rng.exponential(4, n), 0, None)
    late = rng.poisson(1.0, n)

    # --- drift: shift the features ---
    income = income * drift.get("income_scale", 1.0)
    credit = np.clip(credit + drift.get("credit_shift", 0), 300, 850)
    loan = loan * drift.get("loan_scale", 1.0)
    late = late + rng.poisson(drift.get("late_extra", 0.0), n)

    return pd.DataFrame({"age": age, "income": income, "credit_score": credit,
                         "loan_amount": loan, "employment_years": emp,
                         "late_payments": late})


def _make_label(df, rng):
    """The 'true rule' for default, plus noise."""
    lti = df["loan_amount"] / df["income"]                  # loan-to-income ratio
    logit = (-1.0
             - 0.012 * (df["credit_score"] - 650)           # lower credit score -> more risk
             + 0.45 * df["late_payments"]                   # more late payments -> more risk
             + 1.8 * (lti - 1.0)                            # higher loan/income -> more risk
             - 0.07 * df["employment_years"]                # less employment -> more risk
             + rng.normal(0, 1.3, len(df)))                 # noise
    p = 1 / (1 + np.exp(-logit))                            # convert score to probability
    return (rng.random(len(df)) < p).astype(int)


def _apply_corruption(df, corrupt):
    df = df.copy()
    if "income_scale" in corrupt:                           # unit change (rupees -> thousands)
        df["income"] = df["income"] * corrupt["income_scale"]
    if "credit_null_rate" in corrupt:                       # spike of missing values
        mask = np.random.default_rng(0).random(len(df)) < corrupt["credit_null_rate"]
        df.loc[mask, "credit_score"] = np.nan
    return df


def generate_data(n, seed, drift=None, corrupt=None):
    """
    n       : number of rows
    seed    : same seed -> same data (reproducible)
    drift   : dict, e.g. {"income_scale": 0.6}   -> real data drift
    corrupt : dict, e.g. {"income_scale": 0.001} -> pipeline bug
    returns : DataFrame (6 features + 'default')
    """
    rng = np.random.default_rng(seed)
    df = _make_features(n, rng, drift)
    df["default"] = _make_label(df, rng)     # label is created FIRST
    if corrupt:
        df = _apply_corruption(df, corrupt)  # corruption comes AFTER, so the label stays correct
    return df


if __name__ == "__main__":
    d = generate_data(20000, seed=1)
    print(d.describe().round(2))
    print("default rate:", round(d["default"].mean(), 3))