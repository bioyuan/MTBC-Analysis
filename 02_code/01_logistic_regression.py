"""Manuscript-aligned reimplementation. Patient-level input is private."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

COVARIATES = {"crude": [], "adjusted": ["age", "sex", "province"], "sex": ["age", "province"], "age": ["sex", "province"], "region": ["age", "sex"]}

def fit_model(frame, pathogen, mode):
    cols = ["mtbc", pathogen] + COVARIATES[mode]
    d = frame[cols].dropna().copy()
    if d.empty or d[pathogen].nunique() < 2 or d.mtbc.nunique() < 2:
        return {"status": "insufficient_variation", "n": len(d)}
    for name in ["mtbc", pathogen]:
        if not d[name].isin([0, 1]).all():
            raise ValueError(f"{name} must be binary 0/1")
    if mode in ("sex", "age", "region") and d[pathogen].sum() <= 5:
        return {"status": "excluded_positive_count_le_5", "n": len(d)}
    x = d[["mtbc"] + COVARIATES[mode]].copy()
    categorical = [c for c in ("sex", "province") if c in x.columns]
    x = pd.get_dummies(x, columns=categorical, drop_first=True, dtype=float)
    x = sm.add_constant(x.astype(float), has_constant="add")
    try:
        model = sm.Logit(d[pathogen].astype(float), x).fit(disp=False, maxiter=500)
        if not model.mle_retvals.get("converged", False):
            return {"status": "not_converged", "n": len(d)}
        beta, se = model.params["mtbc"], model.bse["mtbc"]
        return {"status": "ok", "n": len(d), "positive_cases": int(d[pathogen].sum()), "or": np.exp(beta), "ci_lower": np.exp(beta - 1.96 * se), "ci_upper": np.exp(beta + 1.96 * se), "pvalue": model.pvalues["mtbc"]}
    except (ValueError, np.linalg.LinAlgError) as exc:
        return {"status": "model_error", "error": str(exc), "n": len(d)}

def run(frame, pathogens):
    required = {"mtbc", "age", "sex", "province", "region"} | set(pathogens)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["age_group"] = pd.cut(frame.age, bins=[0, 15, 45, 65, np.inf], right=False, labels=["0-14", "15-44", "45-64", "65+"])
    rows = []
    for pathogen in pathogens:
        for mode in COVARIATES:
            if mode in ("crude", "adjusted"):
                groups = [("overall", frame)]
            else:
                field = {"age": "age_group", "sex": "sex", "region": "region"}[mode]
                groups = frame.groupby(field, observed=True, sort=True, dropna=True)
            for group, subset in groups:
                rows.append({"pathogen": pathogen, "model": mode, "group": str(group), **fit_model(subset, pathogen, mode)})
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Private individual-level CSV")
    parser.add_argument("--pathogens", required=True, help="Comma-separated outcome column names")
    parser.add_argument("--output", required=True, help="Output CSV path")
    a = parser.parse_args()
    output = Path(a.output); output.parent.mkdir(parents=True, exist_ok=True)
    run(pd.read_csv(a.input), [x.strip() for x in a.pathogens.split(",") if x.strip()]).to_csv(output, index=False)
if __name__ == "__main__":
    main()
