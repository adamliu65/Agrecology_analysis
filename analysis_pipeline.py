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
        )
        for col in df.columns
    }
    return df.rename(columns=rename_map)


def load_data(uploaded_file: Any, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Load csv/xlsx and sanitize column names."""
    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
    return sanitize_columns(df)


def load_data_from_path(path: str | Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Load data from local file path."""
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


def _prepare_model_df(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    model_df = df[columns].dropna().copy()
    for col in columns[1:]:
        model_df[col] = model_df[col].astype("category")
    return model_df


def normality_checks(df: pd.DataFrame, response: str, group: Optional[str] = None) -> pd.DataFrame:
    """Shapiro-Wilk tests globally or by group."""
    out = []

    if group is None:
        vals = df[response].dropna()
        if len(vals) < 3:
            out.append({"group": "ALL", "n": len(vals), "stat": np.nan, "p_value": np.nan})
        else:
            stat, p = stats.shapiro(vals)
            out.append({"group": "ALL", "n": len(vals), "stat": stat, "p_value": p})
    else:
        for g, sub in df.groupby(group, observed=False):
            vals = sub[response].dropna()
            if len(vals) < 3:
                out.append({"group": g, "n": len(vals), "stat": np.nan, "p_value": np.nan})
                continue
            stat, p = stats.shapiro(vals)
            out.append({"group": g, "n": len(vals), "stat": stat, "p_value": p})

    result = pd.DataFrame(out).sort_values("group")
    result["normal_at_0.05"] = result["p_value"] > 0.05
    return result


def levene_homogeneity(df: pd.DataFrame, response: str, group: str) -> pd.DataFrame:
    grouped = [g[response].dropna().values for _, g in df.groupby(group, observed=False)]
    grouped = [x for x in grouped if len(x) > 0]
    if len(grouped) < 2:
        return pd.DataFrame([{"test": "Levene", "stat": np.nan, "p_value": np.nan, "homogeneous_at_0.05": np.nan}])

    stat, p = stats.levene(*grouped, center="median")
    return pd.DataFrame([{"test": "Levene", "stat": stat, "p_value": p, "homogeneous_at_0.05": p > 0.05}])


def anova_analysis(df: pd.DataFrame, response: str, factors: list[str], typ: int = 2) -> pd.DataFrame:
    if not factors:
        raise ValueError("At least one factor is required for ANOVA")

    model_df = _prepare_model_df(df, [response] + factors)
    formula = f"{response} ~ " + " * ".join([f"C({f})" for f in factors])
    model = ols(formula, data=model_df).fit()
    table = anova_lm(model, typ=typ).reset_index().rename(columns={"index": "term"})
    return table


def nested_anova(df: pd.DataFrame, response: str, parent_factor: str, nested_factor: str, typ: int = 2) -> pd.DataFrame:
    model_df = _prepare_model_df(df, [response, parent_factor, nested_factor])
    formula = f"{response} ~ C({parent_factor})/C({nested_factor})"
    model = ols(formula, data=model_df).fit()
    table = anova_lm(model, typ=typ).reset_index().rename(columns={"index": "term"})
    return table


def lsd_posthoc(df: pd.DataFrame, response: str, group: str) -> pd.DataFrame:
    """Fisher's LSD via unadjusted pairwise t-tests."""
    model_df = df[[response, group]].dropna().copy()
    levels = sorted(model_df[group].astype(str).unique())

    rows = []
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
                "p_value": p,
                "significant_at_0.05": p < 0.05,
            }
        )

    return pd.DataFrame(rows)


def dunn_posthoc(df: pd.DataFrame, response: str, group: str, p_adjust: str = "bonferroni") -> pd.DataFrame:
    model_df = df[[response, group]].dropna().copy()
    model_df[group] = model_df[group].astype(str)
    matrix = sp.posthoc_dunn(model_df, val_col=response, group_col=group, p_adjust=p_adjust)
    matrix.index.name = "group"
    return matrix.reset_index()


def kruskal_wallis(df: pd.DataFrame, response: str, group: str) -> pd.DataFrame:
    grouped = [g[response].dropna().values for _, g in df.groupby(group, observed=False)]
    grouped = [x for x in grouped if len(x) > 0]
    if len(grouped) < 2:
        return pd.DataFrame([{"test": "Kruskal-Wallis", "stat": np.nan, "p_value": np.nan}])

    stat, p = stats.kruskal(*grouped)
    return pd.DataFrame([{"test": "Kruskal-Wallis", "stat": stat, "p_value": p}])


def correlation_table(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number])
    return numeric.corr(method=method)
