import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import joblib

from typing import Dict, Any, List
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_recall_fscore_support,
    accuracy_score, confusion_matrix, brier_score_loss
)

SEED = 42
ZERO_AS_MISSING = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
MODEL_PATH = Path("diabetes_model.joblib")
REPORT_PATH = Path("diabetes_report.json")

class ZeroToNaN(BaseEstimator, TransformerMixin):
    def __init__(self, cols: List[str]): self.cols = cols
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        for c in self.cols:
            if c in X.columns:
                X.loc[X[c] == 0, c] = np.nan
        return X

def _build_pipe(model):
    return Pipeline([
        ("zero2nan", ZeroToNaN(ZERO_AS_MISSING)),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", model),
    ])

def _load_data(csv_path="diabetes.csv"):
    df = pd.read_csv(csv_path)
    if "Outcome" not in df.columns:
        raise ValueError("CSV must contain 'Outcome' column.")
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"].astype(int)
    return X, y

def _select_train(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    candidates = {
        "logreg": LogisticRegression(max_iter=4000, class_weight="balanced", random_state=SEED),
        "rf": RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1, class_weight="balanced"),
    }
    # Try adding XGBoost if installed
    try:
        from xgboost import XGBClassifier
        candidates["xgb"] = XGBClassifier(
            n_estimators=600, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=SEED, n_jobs=-1
        )
    except Exception:
        pass

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_auc = {}
    best_name, best_score = None, -1
    for name, model in candidates.items():
        p = _build_pipe(model)
        score = cross_val_score(p, Xtr, ytr, cv=skf, scoring="roc_auc", n_jobs=-1).mean()
        cv_auc[name] = float(score)
        if score > best_score:
            best_score, best_name = score, name

    best_model = clone(candidates[best_name])
    pipe = _build_pipe(best_model).fit(Xtr, ytr)

    # Tune threshold on holdout
    proba = pipe.predict_proba(Xte)[:, 1]
    ts = np.linspace(0.2, 0.8, 61)
    f1s = [f1_score(yte, (proba >= t).astype(int)) for t in ts]
    best_t = float(ts[int(np.argmax(f1s))])

    # Calibrate probabilities
    Xtr_prep = pipe.named_steps["scale"].fit_transform(
        pipe.named_steps["impute"].fit_transform(
            pipe.named_steps["zero2nan"].fit_transform(Xtr)
        )
    )
    cal = CalibratedClassifierCV(pipe.named_steps["model"], method="isotonic", cv=5)
    cal.fit(Xtr_prep, ytr)

    final_pipe = Pipeline([
        ("zero2nan", pipe.named_steps["zero2nan"]),
        ("impute", pipe.named_steps["impute"]),
        ("scale", pipe.named_steps["scale"]),
        ("calibrated_model", cal),
    ])

    # Final metrics
    proba_cal = final_pipe.predict_proba(Xte)[:, 1]
    pred = (proba_cal >= best_t).astype(int)
    acc = accuracy_score(yte, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(yte, pred, average="binary")
    auc = roc_auc_score(yte, proba_cal)
    bri = brier_score_loss(yte, proba_cal)
    cm = confusion_matrix(yte, pred).tolist()

    report = {
        "ACC": float(acc), "PREC": float(prec), "REC": float(rec), "F1": float(f1),
        "ROC_AUC": float(auc), "BRIER": float(bri), "CONFUSION_MATRIX": cm,
        "CV_AUC": cv_auc, "BEST_MODEL": best_name, "BEST_THRESHOLD": best_t
    }
    return final_pipe, report

def load_or_train_bundle():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    X, y = _load_data("diabetes.csv")
    model, report = _select_train(X, y)
    bundle = {"pipeline": model, "threshold": report["BEST_THRESHOLD"]}
    joblib.dump(bundle, MODEL_PATH)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return bundle

def predict_percent(bundle, features: Dict[str, Any]) -> float:
    import pandas as pd
    X = pd.DataFrame([features])
    p = float(bundle["pipeline"].predict_proba(X)[:, 1][0])
    return round(p * 100.0, 2)
