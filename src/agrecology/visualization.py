"""
Visualization utilities for statistical plots and figures.
"""

from __future__ import annotations

import re
from itertools import permutations
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from .constants import (
    DEFAULT_ALPHA,
    PCA_BIPLOT_HEIGHT,
    PCA_POINT_SIZE,
    PCA_RATIO_THRESHOLD,
    PCA_SKEW_THRESHOLD,
    QQPLOT_HEIGHT,
)


def qqplot_figure(
    df: pd.DataFrame,
    response: str,
    group: Optional[str] = None
) -> Optional[go.Figure]:
    """
    Create a Q-Q plot for normality assessment.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    response : str
        Numeric response column name.
    group : Optional[str]
        Grouping column. If None, plots overall distribution.
    
    Returns
    -------
    Optional[go.Figure]
        Plotly figure or None if insufficient data.
    """
    fig = go.Figure()

    def _qq_standardized(values: np.ndarray) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Compute standardized Q-Q plot coordinates."""
        vals = np.asarray(values, dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) < 3:
            return None, None
        std = np.std(vals, ddof=1)
        if std == 0:
            return None, None
        osm, osr = stats.probplot(vals, dist="norm", fit=False)
        osr_std = (np.asarray(osr) - np.mean(vals)) / std
        return np.asarray(osm), osr_std

    x_ranges = []
    if group is None:
        vals = pd.to_numeric(df[response], errors="coerce").values
        osm, osr_std = _qq_standardized(vals)
        if osm is None:
            return None
        x_ranges.append((osm.min(), osm.max()))
        fig.add_trace(go.Scatter(x=osm, y=osr_std, mode="markers", name="ALL"))
    else:
        group_count = 0
        for g, sub in df.groupby(group, observed=False):
            vals = pd.to_numeric(sub[response], errors="coerce").values
            osm, osr_std = _qq_standardized(vals)
            if osm is None:
                continue
            group_count += 1
            x_ranges.append((osm.min(), osm.max()))
            g_name = str(g)
            fig.add_trace(go.Scatter(x=osm, y=osr_std, mode="markers", name=g_name))
        if group_count == 0:
            return None

    x_min = min(v[0] for v in x_ranges)
    x_max = max(v[1] for v in x_ranges)
    line_x = np.linspace(x_min, x_max, 200)
    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_x,
            mode="lines",
            name="y = x",
            line=dict(color="black", dash="dash"),
        )
    )

    fig.update_layout(
        title=f"{response} QQ Plot",
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Standardized Sample Quantiles",
        legend_title=group if group else "Group",
        height=QQPLOT_HEIGHT,
        template="simple_white",
    )
    return fig


def apply_paper_layout(
    fig: go.Figure,
    title: str,
    x_title: str,
    y_title: str,
    height: int = 560,
    width: int = 900,
) -> go.Figure:
    """
    Apply consistent publication-ready styling to a Plotly figure.
    
    Parameters
    ----------
    fig : go.Figure
        Input Plotly figure.
    title : str
        Plot title.
    x_title : str
        X-axis title.
    y_title : str
        Y-axis title.
    height : int, default=560
        Figure height in pixels.
    width : int, default=900
        Figure width in pixels.
    
    Returns
    -------
    go.Figure
        Styled figure.
    """
    fig.update_layout(
        template="simple_white",
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=18)),
        font=dict(size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(l=70, r=30, t=85, b=70),
        height=height,
        width=width,
    )
    fig.update_xaxes(
        title=x_title,
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        ticks="outside",
    )
    fig.update_yaxes(
        title=y_title,
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        ticks="outside",
    )
    return fig


def pca_biplot_2d(
    df: pd.DataFrame,
    columns: list[str],
    color_col: Optional[str] = None,
    label_col: Optional[str] = None,
) -> tuple[Optional[go.Figure], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
    """
    Create a 2D PCA biplot with automatic long-tail handling.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    columns : list[str]
        Numeric columns to include in PCA.
    color_col : Optional[str]
        Column for point coloring (grouping).
    label_col : Optional[str]
        Column for point labels.
    
    Returns
    -------
    tuple[Optional[go.Figure], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]
        (figure, axis_table, vector_table, error_message)
        - figure: Plotly figure or None
        - axis_table: DataFrame with explained variance
        - vector_table: DataFrame with loading and contribution info
        - error_message: Error description if computation failed
    """
    use_cols = [c for c in columns if c in df.columns]
    if len(use_cols) < 2:
        return None, None, None, "PCA requires at least 2 numeric columns."

    x = df[use_cols].apply(pd.to_numeric, errors="coerce")
    valid_idx = x.dropna().index
    x = x.loc[valid_idx]
    if len(x) < 3:
        return None, None, None, "PCA requires at least 3 complete observations."

    # Detect and handle long-tail distributions
    transform_rows: list[dict[str, str | float]] = []
    x_proc = x.copy()
    for col in x_proc.columns:
        ser = x_proc[col].astype(float)
        q50 = float(ser.quantile(0.5))
        q95 = float(ser.quantile(0.95))
        skew = float(ser.skew())
        long_tail = bool(
            (abs(skew) >= PCA_SKEW_THRESHOLD) and (q50 != 0) and ((q95 / q50) >= PCA_RATIO_THRESHOLD)
        )

        transform = "standardize_only"
        note = ""
        if long_tail:
            if (ser > 0).all():
                x_proc[col] = np.log10(ser)
                transform = "log10_then_standardize"
                note = "long-tail detected"
            else:
                transform = "standardize_only"
                note = "long-tail detected but non-positive values; skip log10"

        transform_rows.append({
            "Variable": col,
            "Skewness": skew,
            "Q95/Q50": (q95 / q50) if q50 != 0 else np.nan,
            "Transform": transform,
            "Note": note,
        })

    # Standardize
    std = x_proc.std(ddof=0).replace(0, np.nan)
    z = (x_proc - x_proc.mean()) / std
    z = z.dropna(axis=1)
    if z.shape[1] < 2:
        return None, None, None, "Insufficient variables for PCA (possible constant columns)."

    # SVD
    m = z.to_numpy()
    u, s, vt = np.linalg.svd(m, full_matrices=False)
    eigvals = (s**2) / (m.shape[0] - 1)
    var_ratio = eigvals / eigvals.sum()

    scores_all = u * s
    scores = scores_all[:, :2]
    cos2 = (scores[:, 0] ** 2 + scores[:, 1] ** 2) / np.maximum((scores_all**2).sum(axis=1), 1e-12)

    loadings = vt.T[:, :2] * np.sqrt(eigvals[:2])
    score_lim = float(np.max(np.abs(scores))) if scores.size else 1.0
    vec_lim = float(np.max(np.abs(loadings))) if loadings.size else 1.0
    vec_scale = (score_lim * 0.85 / vec_lim) if vec_lim > 0 else 1.0
    vectors = loadings * vec_scale

    pca_df = pd.DataFrame({"PC1": scores[:, 0], "PC2": scores[:, 1], "cos2": cos2}, index=z.index)
    if color_col and color_col in df.columns:
        pca_df[color_col] = df.loc[pca_df.index, color_col].astype(str)
    if label_col and label_col in df.columns:
        pca_df["label"] = df.loc[pca_df.index, label_col].astype(str)
    else:
        pca_df["label"] = pca_df.index.astype(str)

    # Plot samples
    fig = go.Figure()
    if color_col and color_col in pca_df.columns:
        for g, sub in pca_df.groupby(color_col, observed=False):
            fig.add_trace(
                go.Scatter(
                    x=sub["PC1"],
                    y=sub["PC2"],
                    mode="markers+text",
                    text=sub["label"],
                    textposition="top center",
                    marker=dict(size=PCA_POINT_SIZE, line=dict(color="black", width=1), opacity=0.85),
                    name=str(g),
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=pca_df["PC1"],
                y=pca_df["PC2"],
                mode="markers+text",
                text=pca_df["label"],
                textposition="top center",
                marker=dict(
                    size=PCA_POINT_SIZE,
                    line=dict(color="black", width=1),
                    opacity=0.85,
                    color="white"
                ),
                name="Samples",
            )
        )

    # Plot loading vectors
    for i, col in enumerate(z.columns):
        x_end = float(vectors[i, 0])
        y_end = float(vectors[i, 1])
        fig.add_annotation(
            x=x_end,
            y=y_end,
            ax=0,
            ay=0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            text="",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.2,
            arrowwidth=2,
            arrowcolor="#2F6FA6",
        )
        fig.add_trace(
            go.Scatter(
                x=[x_end],
                y=[y_end],
                mode="text",
                text=[str(col)],
                textposition="top center",
                textfont=dict(color="#C75000", size=14),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Adjust axes
    all_x = np.concatenate([pca_df["PC1"].to_numpy(dtype=float), vectors[:, 0]])
    all_y = np.concatenate([pca_df["PC2"].to_numpy(dtype=float), vectors[:, 1]])
    x_span = max(float(all_x.max() - all_x.min()), 1e-6)
    y_span = max(float(all_y.max() - all_y.min()), 1e-6)
    x_pad = x_span * 0.18
    y_pad = y_span * 0.22
    fig.update_xaxes(range=[float(all_x.min() - x_pad), float(all_x.max() + x_pad)])
    fig.update_yaxes(range=[float(all_y.min() - y_pad), float(all_y.max() + y_pad)])

    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.add_vline(x=0, line_dash="dash", line_color="black")

    fig = apply_paper_layout(
        fig,
        title="PCA - Biplot",
        x_title=f"Dim1 ({var_ratio[0] * 100:.1f}%)",
        y_title=f"Dim2 ({var_ratio[1] * 100:.1f}%)",
        height=PCA_BIPLOT_HEIGHT,
        width=860,
    )

    # Summary tables
    transform_df = pd.DataFrame(transform_rows)

    vector_table = pd.DataFrame({
        "Variable": z.columns,
        "Loading_PC1": loadings[:, 0],
        "Loading_PC2": loadings[:, 1],
        "Vector_X": vectors[:, 0],
        "Vector_Y": vectors[:, 1],
        "Contribution_PC1_%": (loadings[:, 0] ** 2) / np.maximum((loadings[:, 0] ** 2).sum(), 1e-12) * 100,
        "Contribution_PC2_%": (loadings[:, 1] ** 2) / np.maximum((loadings[:, 1] ** 2).sum(), 1e-12) * 100,
    }).sort_values("Contribution_PC1_%", ascending=False)
    vector_table = vector_table.merge(
        transform_df[["Variable", "Transform", "Note"]],
        on="Variable",
        how="left"
    )

    axis_table = pd.DataFrame({
        "Axis": ["Dim1", "Dim2"],
        "Explained_variance_%": [var_ratio[0] * 100, var_ratio[1] * 100],
    })

    return fig, axis_table, vector_table, None
