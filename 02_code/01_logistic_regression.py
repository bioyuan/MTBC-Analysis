"""Manuscript-aligned reimplementation. Patient-level input is private."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

COVARIATES = {"crude": [], "adjusted": ["age", "sex", "province"], "sex": ["age", "province"], "age": ["sex", "province"], "region": ["age", "sex"]}


# Each entry contains the English province name and the corresponding source label.
# Unicode escapes preserve support for source labels in an English-language file.
REGION_PROVINCES = {
    "North": (
        ("Heilongjiang", "\u9ed1\u9f99\u6c5f\u7701"),
        ("Jilin", "\u5409\u6797\u7701"),
        ("Liaoning", "\u8fbd\u5b81\u7701"),
        ("Beijing", "\u5317\u4eac\u5e02"),
        ("Tianjin", "\u5929\u6d25\u5e02"),
        ("Hebei", "\u6cb3\u5317\u7701"),
        ("Shanxi", "\u5c71\u897f\u7701"),
        ("Inner Mongolia", "\u5185\u8499\u53e4\u81ea\u6cbb\u533a"),
        ("Shaanxi", "\u9655\u897f\u7701"),
        ("Gansu", "\u7518\u8083\u7701"),
        ("Ningxia", "\u5b81\u590f\u56de\u65cf\u81ea\u6cbb\u533a"),
        ("Qinghai", "\u9752\u6d77\u7701"),
        ("Xinjiang", "\u65b0\u7586\u7ef4\u543e\u5c14\u81ea\u6cbb\u533a"),
    ),
    "East": (
        ("Shanghai", "\u4e0a\u6d77\u5e02"),
        ("Jiangsu", "\u6c5f\u82cf\u7701"),
        ("Zhejiang", "\u6d59\u6c5f\u7701"),
        ("Anhui", "\u5b89\u5fbd\u7701"),
        ("Fujian", "\u798f\u5efa\u7701"),
        ("Jiangxi", "\u6c5f\u897f\u7701"),
        ("Shandong", "\u5c71\u4e1c\u7701"),
    ),
    "Central-Southern": (
        ("Guangdong", "\u5e7f\u4e1c\u7701"),
        ("Guangxi", "\u5e7f\u897f\u58ee\u65cf\u81ea\u6cbb\u533a"),
        ("Hainan", "\u6d77\u5357\u7701"),
        ("Henan", "\u6cb3\u5357\u7701"),
        ("Hubei", "\u6e56\u5317\u7701"),
        ("Hunan", "\u6e56\u5357\u7701"),
    ),
    "Southwest": (
        ("Chongqing", "\u91cd\u5e86\u5e02"),
        ("Sichuan", "\u56db\u5ddd\u7701"),
        ("Guizhou", "\u8d35\u5dde\u7701"),
        ("Yunnan", "\u4e91\u5357\u7701"),
        ("Tibet", "\u897f\u85cf\u81ea\u6cbb\u533a"),
    ),
}
REGION_BY_PROVINCE = {
    name.casefold(): region
    for region, provinces in REGION_PROVINCES.items()
    for names in provinces
    for name in names
}


def derive_region(province):
    """Map province labels to study regions and keep missing labels unassigned."""
    labels = province.astype("string").str.strip().str.casefold()
    regions = labels.map(REGION_BY_PROVINCE)
    unknown = labels[labels.notna() & labels.ne("") & regions.isna()]
    if not unknown.empty:
        names = ", ".join(sorted(unknown.unique()))
        raise ValueError(f"Unrecognized province labels: {names}")
    return regions


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
    required = {"mtbc", "age", "sex", "province"} | set(pathogens)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["region"] = derive_region(frame["province"])
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
