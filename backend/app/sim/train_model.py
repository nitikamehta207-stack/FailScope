"""
train_model.py - trains model v1 (good) and model v2 (a bad candidate for the
"bad deploy" fault). Saves both to backend/models/.
Run from the FailScope folder:  python -m backend.app.sim.train_model
"""
from pathlib import Path
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from .dataset import generate_data, FEATURES

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def train_v1():
    train = generate_data(20000, seed=1)    # training data
    test = generate_data(5000, seed=2)      # different seed -> model has never seen this
    model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=0)
    model.fit(train[FEATURES], train["default"])
    return model, test


def train_bad_v2():
    # v2 was trained with a feature-encoding mismatch: employment_years was
    # stored as "years until retirement" (40 - years). It looks fine in
    # training, but it misreads normal production data.
    train = generate_data(20000, seed=11)
    train["employment_years"] = 40 - train["employment_years"]
    model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=0)
    model.fit(train[FEATURES], train["default"])
    return model


def evaluate(model, df):
    pred = model.predict(df[FEATURES])
    proba = model.predict_proba(df[FEATURES])[:, 1]
    return accuracy_score(df["default"], pred), roc_auc_score(df["default"], proba)


if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model, test = train_v1()
    joblib.dump(model, MODELS_DIR / "model_v1.joblib")
    print("default rate (test):", round(test["default"].mean(), 3))

    acc, auc = evaluate(model, test)
    print(f"v1 normal              acc={acc:.3f} auc={auc:.3f}")

    # Check how v1 behaves under each injected problem
    for name, kw in [("drift income x0.6", dict(drift={"income_scale": 0.6})),
                     ("corrupt income x0.001", dict(corrupt={"income_scale": 0.001})),
                     ("corrupt credit 30% null", dict(corrupt={"credit_null_rate": 0.3}))]:
        d = generate_data(5000, seed=3, **kw)
        d_eval = d.fillna(d[FEATURES].median())   # fill NaNs so sklearn doesn't fail
        a, u = evaluate(model, d_eval)
        print(f"v1 {name:<22} acc={a:.3f} auc={u:.3f}")

    bad = train_bad_v2()
    joblib.dump(bad, MODELS_DIR / "model_v2.joblib")
    a, u = evaluate(bad, test)
    print(f"v2 (bad) on normal data acc={a:.3f} auc={u:.3f}")
    print("saved ->", MODELS_DIR)