"""
Core statistical analysis functions.
"""

from __future__ import annotations

from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

from .constants import DEFAULT_ANOVA_TYPE, DEFAULT_ALPHA, MIN_SAMPLES_FOR_SHAPIRO, MIN_GROUPS_FOR_LEVENE


def _is_deterministic_mapping(a: pd.Series, b: pd.Series) -> bool:
    """Check if column a deterministically maps to column b."""
    tmp = pd.DataFrame({"a": a.astype(str), "b": b.astype(str)}).dropna()
    if tmp.empty:
        return False
    return (tmp.groupby("a")["b"].nunique() <= 1).all()


def _clean_factor_list(df: pd.DataFrame, factors: list[str]) -> list[str]:
    """
    Remove problematic factors (single level, confounded, etc.).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    factors : list[str]
        Factor column names to validate.
    
    Returns
    -------
    list[str]
        Cleaned factor list.
    """
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


def _prepare_model_df(df: pd.DataFrame, response: str, factors: list[str]) -> pd.DataFrame:
    """
    Prepare DataFrame for model fitting (remove NAs, convert factors to categories).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    response : str
        Response variable name.
    factors : list[str]
        Factor column names.
    
    Returns
    -------
    pd.DataFrame
        Cleaned data ready for modeling.
    """
    model_df = df[[response] + factors].dropna().copy()
    for col in factors:
        model_df[col] = model_df[col].astype("category")
    return model_df


def _safe_anova_table(model, typ: int) -> pd.DataFrame:
    """
    Safely compute ANOVA table with fallback to robustness.
    
    Parameters
    ----------
    model
        Fitted OLS model from statsmodels.
    typ : int
        Type of sum of squares (1, 2, or 3).
    
    Returns
    -------
    pd.DataFrame
        ANOVA table.
    """
    try:
        return anova_lm(model, typ=typ)
    except Exception:
        try:
            return anova_lm(model, typ=1)
        except Exception:
            return anova_lm(model, typ=2, robust='hc3')


def normality_checks(
    df: pd.DataFrame,
    response: str,
    group: Optional[str] = None
) -> pd.DataFrame:
    """
    Perform Shapiro-Wilk normality test on response variable.
    
    Can test overall or by group.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    response : str
        Numeric response column name.
    group : Optional[str]
        Grouping column. If None, tests overall distribution.
    
    Returns
    -------
    pd.DataFrame
        Normality test results with columns: group, n, stat, p_value, normal_at_0.05
    """
    out = []
    if group is None:
        vals = pd.to_numeric(df[response], errors="coerce").dropna()
        if len(vals) < MIN_SAMPLES_FOR_SHAPIRO:
            out.append({"group": "ALL", "n": len(vals), "stat": np.nan, "p_value": np.nan})
        else:
            stat, p = stats.shapiro(vals)
            out.append({"group": "ALL", "n": len(vals), "stat": stat, "p_value": p})
    else:
        for g, sub in df.groupby(group, observed=False):
            vals = pd.to_numeric(sub[response], errors="coerce").dropna()
            if len(vals) < MIN_SAMPLES_FOR_SHAPIRO:
                out.append({"group": g, "n": len(vals), "stat": np.nan, "p_value": np.nan})
                continue
            stat, p = stats.shapiro(vals)
            out.append({"group": g, "n": len(vals), "stat": stat, "p_value": p})

    result = pd.DataFrame(out).sort_values("group")
    result["normal_at_0.05"] = result["p_value"] > DEFAULT_ALPHA
    return result


def levene_homogeneity(
    df: pd.DataFrame,
    response: str,
    group: str
) -> pd.DataFrame:
    """
    Perform Levene's test for homogeneity of variance.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    response : str
        Numeric response column name.
    group : str
        Grouping column name.
    
    Returns
    -------
    pd.DataFrame
        Levene test results with columns: test, stat, p_value, homogeneous_at_0.05
    """
    grouped = [
        pd.to_numeric(g[response], errors="coerce").dropna().values
        for _, g in df.groupby(group, observed=False)
    ]
    grouped = [x for x in grouped if len(x) > 0]
    
    if len(grouped) < MIN_GROUPS_FOR_LEVENE:
        return pd.DataFrame([{
            "test": "Levene",
            "stat": np.nan,
            "p_value": np.nan,
            "homogeneous_at_0.05": np.nan
        }])
    
    stat, p = stats.levene(*grouped, center="median")
    return pd.DataFrame([{
        "test": "Levene",
        "stat": stat,
        "p_value": p,
        "homogeneous_at_0.05": p > DEFAULT_ALPHA
    }])


def anova_analysis(
    df: pd.DataFrame,
    response: str,
    factors: list[str],
    typ: int = DEFAULT_ANOVA_TYPE,
    block_factor: Optional[str] = None,
) -> pd.DataFrame:
    """
    Perform factorial ANOVA with optional blocking factor.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    response : str
        Numeric response column name.
    factors : list[str]
        Factor column names (crossed design).
    typ : int, default=2
        Type of sum of squares (1, 2, or 3).
    block_factor : Optional[str]
        Blocking/replication factor (added as main effect).
    
    Returns
    -------
    pd.DataFrame
        ANOVA table with columns: term, sum_sq, df, F, PR(>F) (or similar).
    
    Raises
    ------
    ValueError
        If no valid factors or block factor provided.
    """
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
    typ: int = DEFAULT_ANOVA_TYPE,
    block_factor: Optional[str] = None,
) -> pd.DataFrame:
    """
    Perform nested ANOVA.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    response : str
        Numeric response column name.
    parent_factor : str
        Parent/higher-level factor name.
    nested_factor : str
        Nested/lower-level factor name.
    typ : int, default=2
        Type of sum of squares (1, 2, or 3).
    block_factor : Optional[str]
        Blocking/replication factor.
    
    Returns
    -------
    pd.DataFrame
        ANOVA table.
    
    Raises
    ------
    ValueError
        If parent and nested factors are same, missing, or have < 2 levels.
    """
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


def kruskal_wallis(
    df: pd.DataFrame,
    response: str,
    group: str
) -> pd.DataFrame:
    """
    Perform Kruskal-Wallis H-test (non-parametric alternative to one-way ANOVA).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    response : str
        Numeric response column name.
    group : str
        Grouping column name.
    
    Returns
    -------
    pd.DataFrame
        Test results with columns: test, stat, p_value
    """
    grouped = [
        pd.to_numeric(g[response], errors="coerce").dropna().values
        for _, g in df.groupby(group, observed=False)
    ]
    grouped = [x for x in grouped if len(x) > 0]
    
    if len(grouped) < MIN_GROUPS_FOR_LEVENE:
        return pd.DataFrame([{"test": "Kruskal-Wallis", "stat": np.nan, "p_value": np.nan}])
    
    stat, p = stats.kruskal(*grouped)
    return pd.DataFrame([{"test": "Kruskal-Wallis", "stat": stat, "p_value": p}])


def correlation_table(
    df: pd.DataFrame,
    method: str = "pearson",
    columns: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Compute correlation matrix for numeric columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    method : str, default="pearson"
        Correlation method ("pearson" or "spearman").
    columns : Optional[list[str]]
        Specific columns to use. If None, uses all numeric columns.
    
    Returns
    -------
    pd.DataFrame
        Correlation matrix.
    """
    if columns:
        numeric = df[columns].apply(pd.to_numeric, errors="coerce")
    else:
        numeric = df.select_dtypes(include=[np.number])
    return numeric.corr(method=method)
