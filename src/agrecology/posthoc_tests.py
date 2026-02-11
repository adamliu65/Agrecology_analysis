"""
Post-hoc tests for pairwise comparisons.
"""

from __future__ import annotations

from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

from .constants import DEFAULT_ALPHA, DEFAULT_DUNN_ADJUST


def lsd_posthoc(
    df: pd.DataFrame,
    response: str,
    group: str
) -> pd.DataFrame:
    """
    LSD (Least Significant Difference) post-hoc test.
    
    Performs pairwise t-tests without correction.
    
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
        Pairwise comparison results with columns:
        group_a, group_b, n_a, n_b, mean_a, mean_b, t_stat, p_value, significant_at_0.05
    """
    model_df = df[[response, group]].copy()
    model_df[response] = pd.to_numeric(model_df[response], errors="coerce")
    model_df = model_df.dropna()
    levels = sorted(model_df[group].astype(str).unique())
    
    rows = []
    for a, b in combinations(levels, 2):
        xa = model_df.loc[model_df[group].astype(str) == a, response]
        xb = model_df.loc[model_df[group].astype(str) == b, response]
        stat, p = stats.ttest_ind(xa, xb, equal_var=True)
        rows.append({
            "group_a": a,
            "group_b": b,
            "n_a": len(xa),
            "n_b": len(xb),
            "mean_a": xa.mean(),
            "mean_b": xb.mean(),
            "t_stat": stat,
            "p_value": p,
            "significant_at_0.05": p < DEFAULT_ALPHA
        })
    
    return pd.DataFrame(rows)


def bonferroni_posthoc(
    df: pd.DataFrame,
    response: str,
    group: str
) -> pd.DataFrame:
    """
    Bonferroni-corrected pairwise t-tests.
    
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
        Pairwise comparison results with columns:
        group_a, group_b, n_a, n_b, mean_a, mean_b, t_stat, p_value_raw, p_value_bonferroni, significant_at_0.05
    """
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
        rows.append({
            "group_a": a,
            "group_b": b,
            "n_a": len(xa),
            "n_b": len(xb),
            "mean_a": xa.mean(),
            "mean_b": xb.mean(),
            "t_stat": stat,
            "p_value_raw": p,
        })
        raw_pvals.append(p)

    if not rows:
        return pd.DataFrame(
            columns=["group_a", "group_b", "p_value_raw", "p_value_bonferroni", "significant_at_0.05"]
        )

    _, p_adj, _, _ = multipletests(raw_pvals, alpha=DEFAULT_ALPHA, method="bonferroni")
    out = pd.DataFrame(rows)
    out["p_value_bonferroni"] = p_adj
    out["significant_at_0.05"] = out["p_value_bonferroni"] < DEFAULT_ALPHA
    return out


def tukey_posthoc(
    df: pd.DataFrame,
    response: str,
    group: str
) -> pd.DataFrame:
    """
    Tukey HSD (Honest Significant Difference) post-hoc test.
    
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
        Pairwise comparison results with columns:
        group_a, group_b, mean_diff, p_adj, ci_low, ci_high, reject_at_0.05
    """
    model_df = df[[response, group]].copy()
    model_df[response] = pd.to_numeric(model_df[response], errors="coerce")
    model_df = model_df.dropna()
    model_df[group] = model_df[group].astype(str)
    
    if model_df[group].nunique() < 2:
        return pd.DataFrame(
            columns=["group_a", "group_b", "mean_diff", "p_adj", "ci_low", "ci_high", "reject_at_0.05"]
        )

    result = pairwise_tukeyhsd(endog=model_df[response], groups=model_df[group], alpha=DEFAULT_ALPHA)
    table = pd.DataFrame(result._results_table.data[1:], columns=result._results_table.data[0])
    table = table.rename(columns={
        "group1": "group_a",
        "group2": "group_b",
        "meandiff": "mean_diff",
        "p-adj": "p_adj",
        "lower": "ci_low",
        "upper": "ci_high",
        "reject": "reject_at_0.05",
    })
    return table


def dunn_posthoc(
    df: pd.DataFrame,
    response: str,
    group: str,
    p_adjust: str = DEFAULT_DUNN_ADJUST
) -> pd.DataFrame:
    """
    Dunn's post-hoc test (non-parametric, for use after Kruskal-Wallis).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    response : str
        Numeric response column name.
    group : str
        Grouping column name.
    p_adjust : str, default="bonferroni"
        P-value adjustment method ("bonferroni", "holm", "fdr_bh", etc.).
    
    Returns
    -------
    pd.DataFrame
        Pairwise comparison matrix (groups × groups).
    """
    model_df = df[[response, group]].copy()
    model_df[response] = pd.to_numeric(model_df[response], errors="coerce")
    model_df = model_df.dropna()
    model_df[group] = model_df[group].astype(str)
    
    matrix = sp.posthoc_dunn(model_df, val_col=response, group_col=group, p_adjust=p_adjust)
    matrix.index.name = "group"
    return matrix.reset_index()
