from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names for formula processing."""
    rename_map = {
        col: (
            str(col)
            .strip()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(":", "_")
            .replace("/", "_")
            .replace("*", "")
        )
        for col in df.columns
    }
    return df.rename(columns=rename_map)


def load_data(uploaded_file: Any, sheet_name: Optional[str] = None) -> pd.DataFrame:
    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
    return sanitize_columns(df)


def load_data_from_path(path: str | Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    else:
        df = pd.read_excel(p, sheet_name=sheet_name)
    return sanitize_columns(df)


def split_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def select_parameter_columns(
    df: pd.DataFrame,
    start_col: Optional[str] = None,
    manual_cols: Optional[list[str]] = None,
    exclude_cols: Optional[list[str]] = None,
    min_numeric_ratio: float = 0.6,
) -> list[str]:
    exclude_set = set(exclude_cols or [])

    if manual_cols:
        return [c for c in manual_cols if c not in exclude_set]

    cols = list(df.columns)
    if start_col and start_col in cols:
        start_idx = cols.index(start_col)
        candidates = [c for c in cols[start_idx:] if c not in exclude_set]
    else:
        candidates = [c for c in cols if c not in exclude_set]

    selected = []
    for col in candidates:
        ser = pd.to_numeric(df[col], errors="coerce")
        ratio = ser.notna().mean()
        if ratio >= min_numeric_ratio:
            selected.append(col)
    return selected


def coerce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _prepare_model_df(df: pd.DataFrame, response: str, factors: list[str]) -> pd.DataFrame:
    model_df = df[[response] + factors].dropna().copy()
    for col in factors:
        model_df[col] = model_df[col].astype("category")
    return model_df




def _is_deterministic_mapping(a: pd.Series, b: pd.Series) -> bool:
    tmp = pd.DataFrame({"a": a.astype(str), "b": b.astype(str)}).dropna()
    if tmp.empty:
        return False
    return (tmp.groupby("a")["b"].nunique() <= 1).all()


def _clean_factor_list(df: pd.DataFrame, factors: list[str]) -> list[str]:
    cleaned: list[str] = []
    for f in factors:
        if f not in df.columns:
            continue
        nun = df[f].dropna().nunique()
        if nun <= 1:
            continue

        confounded = False
        for kept in cleaned:
            if df[f].astype(str).equals(df[kept].astype(str)):
                confounded = True
                break
            if _is_deterministic_mapping(df[f], df[kept]) or _is_deterministic_mapping(df[kept], df[f]):
                confounded = True
                break

        if not confounded:
            cleaned.append(f)
    return cleaned


def _safe_anova_table(model, typ: int) -> pd.DataFrame:
    try:
        return anova_lm(model, typ=typ)
    except Exception:
        try:
            return anova_lm(model, typ=1)
        except Exception:
            return anova_lm(model, typ=2, robust='hc3')

def normality_checks(df: pd.DataFrame, response: str, group: Optional[str] = None) -> pd.DataFrame:
    out = []
    if group is None:
        vals = pd.to_numeric(df[response], errors="coerce").dropna()
        if len(vals) < 3:
            out.append({"group": "ALL", "n": len(vals), "stat": np.nan, "p_value": np.nan})
        else:
            stat, p = stats.shapiro(vals)
            out.append({"group": "ALL", "n": len(vals), "stat": stat, "p_value": p})
    else:
        for g, sub in df.groupby(group, observed=False):
            vals = pd.to_numeric(sub[response], errors="coerce").dropna()
            if len(vals) < 3:
                out.append({"group": g, "n": len(vals), "stat": np.nan, "p_value": np.nan})
                continue
            stat, p = stats.shapiro(vals)
            out.append({"group": g, "n": len(vals), "stat": stat, "p_value": p})

    result = pd.DataFrame(out).sort_values("group")
    result["normal_at_0.05"] = result["p_value"] > 0.05
    return result


def levene_homogeneity(df: pd.DataFrame, response: str, group: str) -> pd.DataFrame:
    grouped = [pd.to_numeric(g[response], errors="coerce").dropna().values for _, g in df.groupby(group, observed=False)]
    grouped = [x for x in grouped if len(x) > 0]
    if len(grouped) < 2:
        return pd.DataFrame([{"test": "Levene", "stat": np.nan, "p_value": np.nan, "homogeneous_at_0.05": np.nan}])
    stat, p = stats.levene(*grouped, center="median")
    return pd.DataFrame([{"test": "Levene", "stat": stat, "p_value": p, "homogeneous_at_0.05": p > 0.05}])


def anova_analysis(
    df: pd.DataFrame,
    response: str,
    factors: list[str],
    typ: int = 2,
    block_factor: Optional[str] = None,
) -> pd.DataFrame:
    factors = _clean_factor_list(df, factors)
    if not factors and not block_factor:
        raise ValueError("At least one valid factor (or block factor) is required for ANOVA")

    model_terms = [f"C({f})" for f in factors]
    formula_rhs = " * ".join(model_terms) if model_terms else "1"
    all_factors = list(factors)

    if block_factor:
        formula_rhs = f"C({block_factor}) + {formula_rhs}" if formula_rhs != "1" else f"C({block_factor})"
        all_factors = [block_factor] + all_factors

    model_df = _prepare_model_df(df, response, all_factors)
    formula = f"{response} ~ {formula_rhs}"
    model = ols(formula, data=model_df).fit()

    table = _safe_anova_table(model, typ=typ)
    if table.empty and factors:
        additive_rhs = " + ".join([f"C({f})" for f in factors])
        if block_factor:
            additive_rhs = f"C({block_factor}) + {additive_rhs}"
        model2 = ols(f"{response} ~ {additive_rhs}", data=model_df).fit()
        table = _safe_anova_table(model2, typ=1)
    return table.reset_index().rename(columns={"index": "term"})


def nested_anova(
    df: pd.DataFrame,
    response: str,
    parent_factor: str,
    nested_factor: str,
    typ: int = 2,
    block_factor: Optional[str] = None,
) -> pd.DataFrame:
    if parent_factor == nested_factor:
        raise ValueError("parent_factor and nested_factor must be different")
    if parent_factor not in df.columns or nested_factor not in df.columns:
        raise ValueError("Nested ANOVA factors must exist in dataframe")
    if df[parent_factor].dropna().nunique() <= 1 or df[nested_factor].dropna().nunique() <= 1:
        raise ValueError("Nested ANOVA factors must have at least two levels")

    all_factors = [parent_factor, nested_factor]
    formula_rhs = f"C({parent_factor})/C({nested_factor})"
    if block_factor:
        formula_rhs = f"C({block_factor}) + {formula_rhs}"
        all_factors = [block_factor] + all_factors

    model_df = _prepare_model_df(df, response, all_factors)
    model = ols(f"{response} ~ {formula_rhs}", data=model_df).fit()
    table = _safe_anova_table(model, typ=typ)
    return table.reset_index().rename(columns={"index": "term"})


def lsd_posthoc(df: pd.DataFrame, response: str, group: str) -> pd.DataFrame:
    model_df = df[[response, group]].copy()
    model_df[response] = pd.to_numeric(model_df[response], errors="coerce")
    model_df = model_df.dropna()
    levels = sorted(model_df[group].astype(str).unique())
    rows = []
    for a, b in combinations(levels, 2):
        xa = model_df.loc[model_df[group].astype(str) == a, response]
        xb = model_df.loc[model_df[group].astype(str) == b, response]
        stat, p = stats.ttest_ind(xa, xb, equal_var=True)
        rows.append({"group_a": a, "group_b": b, "n_a": len(xa), "n_b": len(xb), "mean_a": xa.mean(), "mean_b": xb.mean(), "t_stat": stat, "p_value": p, "significant_at_0.05": p < 0.05})
    return pd.DataFrame(rows)


def bonferroni_posthoc(df: pd.DataFrame, response: str, group: str) -> pd.DataFrame:
    model_df = df[[response, group]].copy()
    model_df[response] = pd.to_numeric(model_df[response], errors="coerce")
    model_df = model_df.dropna()
    levels = sorted(model_df[group].astype(str).unique())

    rows = []
    raw_pvals = []
    for a, b in combinations(levels, 2):
        xa = model_df.loc[model_df[group].astype(str) == a, response]
        xb = model_df.loc[model_df[group].astype(str) == b, response]
        stat, p = stats.ttest_ind(xa, xb, equal_var=True)
        rows.append(
            {
                "group_a": a,
                "group_b": b,
                "n_a": len(xa),
                "n_b": len(xb),
                "mean_a": xa.mean(),
                "mean_b": xb.mean(),
                "t_stat": stat,
                "p_value_raw": p,
            }
        )
        raw_pvals.append(p)

    if not rows:
        return pd.DataFrame(columns=["group_a", "group_b", "p_value_raw", "p_value_bonferroni", "significant_at_0.05"])

    _, p_adj, _, _ = multipletests(raw_pvals, alpha=0.05, method="bonferroni")
    out = pd.DataFrame(rows)
    out["p_value_bonferroni"] = p_adj
    out["significant_at_0.05"] = out["p_value_bonferroni"] < 0.05
    return out


def tukey_posthoc(df: pd.DataFrame, response: str, group: str) -> pd.DataFrame:
    model_df = df[[response, group]].copy()
    model_df[response] = pd.to_numeric(model_df[response], errors="coerce")
    model_df = model_df.dropna()
    model_df[group] = model_df[group].astype(str)
    if model_df[group].nunique() < 2:
        return pd.DataFrame(columns=["group_a", "group_b", "mean_diff", "p_adj", "ci_low", "ci_high", "reject_at_0.05"])

    result = pairwise_tukeyhsd(endog=model_df[response], groups=model_df[group], alpha=0.05)
    table = pd.DataFrame(result._results_table.data[1:], columns=result._results_table.data[0])
    table = table.rename(
        columns={
            "group1": "group_a",
            "group2": "group_b",
            "meandiff": "mean_diff",
            "p-adj": "p_adj",
            "lower": "ci_low",
            "upper": "ci_high",
            "reject": "reject_at_0.05",
        }
    )
    return table


def dunn_posthoc(df: pd.DataFrame, response: str, group: str, p_adjust: str = "bonferroni") -> pd.DataFrame:
    model_df = df[[response, group]].copy()
    model_df[response] = pd.to_numeric(model_df[response], errors="coerce")
    model_df = model_df.dropna()
    model_df[group] = model_df[group].astype(str)
    matrix = sp.posthoc_dunn(model_df, val_col=response, group_col=group, p_adjust=p_adjust)
    matrix.index.name = "group"
    return matrix.reset_index()


def kruskal_wallis(df: pd.DataFrame, response: str, group: str) -> pd.DataFrame:
    grouped = [pd.to_numeric(g[response], errors="coerce").dropna().values for _, g in df.groupby(group, observed=False)]
    grouped = [x for x in grouped if len(x) > 0]
    if len(grouped) < 2:
        return pd.DataFrame([{"test": "Kruskal-Wallis", "stat": np.nan, "p_value": np.nan}])
    stat, p = stats.kruskal(*grouped)
    return pd.DataFrame([{"test": "Kruskal-Wallis", "stat": stat, "p_value": p}])


def correlation_table(df: pd.DataFrame, method: str = "pearson", columns: Optional[list[str]] = None) -> pd.DataFrame:
    if columns:
        numeric = df[columns].apply(pd.to_numeric, errors="coerce")
    else:
        numeric = df.select_dtypes(include=[np.number])
    return numeric.corr(method=method)
