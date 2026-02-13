import io
import re
import sys
from pathlib import Path
from itertools import permutations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

# Ensure `src/agrecology` is importable in Streamlit Cloud/local execution.
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agrecology import (
    anova_analysis,
    bonferroni_posthoc,
    coerce_numeric_columns,
    correlation_table,
    dunn_posthoc,
    kruskal_wallis,
    levene_homogeneity,
    load_data,
    lsd_posthoc,
    nested_anova,
    normality_checks,
    select_parameter_columns,
    split_columns,
    tukey_posthoc,
)


def highlight_significant_rows(table: pd.DataFrame, alpha: float = 0.05):
    p_col = next((c for c in ["PR(>F)", "p_value", "pvalue"] if c in table.columns), None)
    if p_col is None:
        return table

    def _row_style(row: pd.Series) -> list[str]:
        p_val = pd.to_numeric(row.get(p_col), errors="coerce")
        term = str(row.get("term", "")).strip().lower()
        is_sig = pd.notna(p_val) and p_val < alpha and term != "residual"
        row_style = "background-color: #fff3bf;" if is_sig else ""
        return [row_style for _ in row.index]

    return table.style.apply(_row_style, axis=1)


def qqplot_figure(df: pd.DataFrame, response: str, group: str | None = None):
    fig = go.Figure()

    def _qq_standardized(values: np.ndarray):
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
        height=520,
    )
    return fig


def render_multi_button_selector(label: str, options: list[str], key_prefix: str, default: list[str]):
    widget_key = f"{key_prefix}_widget"
    action_key = f"{key_prefix}_action"
    if widget_key not in st.session_state:
        st.session_state[widget_key] = [x for x in default if x in options] or options[:1]
    else:
        # Keep only currently valid options when option list changes.
        st.session_state[widget_key] = [x for x in (st.session_state.get(widget_key) or []) if x in options]

    # Apply quick-select action before creating the widget to avoid Streamlit state mutation errors.
    pending_action = st.session_state.pop(action_key, None)
    if pending_action == "all":
        st.session_state[widget_key] = options.copy()
    elif pending_action == "clear":
        st.session_state[widget_key] = []

    st.markdown(f"**{label}**")

    if hasattr(st, "pills"):
        selected = st.pills(
            " ",
            options=options,
            selection_mode="multi",
            key=widget_key,
        )
    else:
        selected = st.multiselect(
            " ",
            options=options,
            key=widget_key,
        )

    if not options:
        return []

    c1, c2, c3 = st.columns([1, 1, 6])
    if c1.button("全選", key=f"{key_prefix}_select_all"):
        st.session_state[action_key] = "all"
        st.rerun()
    if c2.button("全不選", key=f"{key_prefix}_clear_all"):
        st.session_state[action_key] = "clear"
        st.rerun()
    c3.empty()

    return selected


def _extract_factor_names_from_term(term: str) -> list[str]:
    return re.findall(r"C\(([^)]+)\)", str(term))


def _anova_p_col(table: pd.DataFrame) -> str | None:
    for col in ["PR(>F)", "p_value", "pvalue"]:
        if col in table.columns:
            return col
    return None


def _ordered_levels(values: list[str]) -> list[str]:
    vals = [str(v) for v in values]
    uniq = list(dict.fromkeys(vals))
    if not uniq:
        return uniq

    # Pure numeric labels should follow numeric order (1, 2, 3...) rather than lexicographic order.
    try:
        nums = [float(v) for v in uniq]
        if all(np.isfinite(n) for n in nums):
            return [x for _, x in sorted(zip(nums, uniq), key=lambda t: t[0])]
    except Exception:
        pass

    # Prefer natural ordering like T1, T2, ... T8 when pattern is consistent.
    m = [re.match(r"^([A-Za-z]+)(\d+)$", v) for v in uniq]
    if all(x is not None for x in m):
        prefixes = {x.group(1) for x in m if x is not None}
        if len(prefixes) == 1:
            return sorted(uniq, key=lambda s: int(re.match(r"^[A-Za-z]+(\d+)$", s).group(1)))

    if all(re.match(r"^T\d+$", v) for v in uniq):
        return sorted(uniq, key=lambda s: int(s[1:]))

    return sorted(uniq)


def _format_response_y_label(response: str, unit_row: pd.Series | None = None, with_mean: bool = False) -> str:
    if unit_row is None:
        return f"{response} (mean)" if with_mean else response

    unit_val = unit_row.get(response, None)
    if pd.isna(unit_val):
        return f"{response} (mean)" if with_mean else response

    unit_txt = str(unit_val).strip()
    if not unit_txt or unit_txt.lower() in {"nan", "none", "na", "-", "--"}:
        return f"{response} (mean)" if with_mean else response

    if with_mean:
        return f"{response} (mean, {unit_txt})"
    return f"{response} ({unit_txt})"


def _normalize_factor_label(v) -> str | None:
    if pd.isna(v):
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        n = float(s)
        if np.isfinite(n):
            if n.is_integer():
                return str(int(n))
            return str(n)
    except Exception:
        pass
    return s


def _pairwise_significance_for_cld(
    df: pd.DataFrame,
    response: str,
    group: str,
    method: str,
    dunn_adjust: str = "bonferroni",
) -> dict[tuple[str, str], bool]:
    levels = _ordered_levels([x for x in df[group].map(_normalize_factor_label).dropna().tolist()])
    sig = {(a, b): False for a in levels for b in levels}

    try:
        if method == "LSD":
            ph = lsd_posthoc(df, response=response, group=group)
            for _, row in ph.iterrows():
                a = _normalize_factor_label(row.get("group_a"))
                b = _normalize_factor_label(row.get("group_b"))
                rej = bool(row.get("significant_at_0.05", False))
                if a is not None and b is not None and a in levels and b in levels:
                    sig[(a, b)] = rej
                    sig[(b, a)] = rej
        elif method == "Tukey":
            ph = tukey_posthoc(df, response=response, group=group)
            for _, row in ph.iterrows():
                a = _normalize_factor_label(row.get("group_a"))
                b = _normalize_factor_label(row.get("group_b"))
                rej = row.get("reject_at_0.05", False)
                if isinstance(rej, str):
                    rej = rej.strip().lower() == "true"
                if a is not None and b is not None and a in levels and b in levels:
                    sig[(a, b)] = bool(rej)
                    sig[(b, a)] = bool(rej)
        elif method == "Bonferroni":
            ph = bonferroni_posthoc(df, response=response, group=group)
            for _, row in ph.iterrows():
                a = _normalize_factor_label(row.get("group_a"))
                b = _normalize_factor_label(row.get("group_b"))
                rej = bool(row.get("significant_at_0.05", False))
                if a is not None and b is not None and a in levels and b in levels:
                    sig[(a, b)] = rej
                    sig[(b, a)] = rej
        elif method == "Dunn":
            ph = dunn_posthoc(df, response=response, group=group, p_adjust=dunn_adjust)
            ph = ph.copy()
            if "group" in ph.columns:
                ph["group"] = ph["group"].map(_normalize_factor_label)
                norm_col_map = {c: (_normalize_factor_label(c) if c != "group" else "group") for c in ph.columns}
                for _, row in ph.iterrows():
                    a = row["group"]
                    if a is None:
                        continue
                    for col in ph.columns:
                        if col == "group":
                            continue
                        b = norm_col_map[col]
                        if b is None or b not in levels:
                            continue
                        p = pd.to_numeric(row.get(col), errors="coerce")
                        if pd.notna(p):
                            rej = bool(p < 0.05)
                            sig[(a, b)] = rej
                            sig[(b, a)] = rej
    except Exception:
        # Keep conservative default (no significant differences) if post-hoc fails.
        pass

    return sig


def _p_to_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p <= 0.001:
        return "***"
    if p <= 0.01:
        return "**"
    if p <= 0.05:
        return "*"
    return ""


def _correlation_pvalue_matrix(df: pd.DataFrame, columns: list[str], method: str = "pearson") -> pd.DataFrame:
    num = df[columns].apply(pd.to_numeric, errors="coerce")
    cols = list(num.columns)
    pmat = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)

    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i == j:
                pmat.loc[c1, c2] = 0.0
                continue
            pair = num[[c1, c2]].dropna()
            if len(pair) < 3:
                continue
            try:
                if method == "spearman":
                    _, p = stats.spearmanr(pair[c1], pair[c2])
                else:
                    _, p = stats.pearsonr(pair[c1], pair[c2])
                pmat.loc[c1, c2] = p
            except Exception:
                pmat.loc[c1, c2] = np.nan
    return pmat


def _make_cld_from_significance(sig: dict[tuple[str, str], bool], group_order: list[str]) -> dict[str, str]:
    levels = [str(x) for x in group_order]
    if not levels:
        return {}

    def _reduce_sets(sets_in: list[set[str]]) -> list[set[str]]:
        unique = []
        seen = set()
        for s in sets_in:
            fs = frozenset(s)
            if fs and fs not in seen:
                seen.add(fs)
                unique.append(set(s))

        # Absorb redundant sets (strict subset of another set).
        out = []
        for i, s in enumerate(unique):
            if any(i != j and s < t for j, t in enumerate(unique)):
                continue
            out.append(s)
        return out

    ns_pairs = [(a, b) for i, a in enumerate(levels) for b in levels[i + 1 :] if not sig.get((a, b), False)]

    def _is_valid_cover(sets_in: list[set[str]]) -> bool:
        if not sets_in:
            return False
        # Every group must own at least one letter.
        for g in levels:
            if not any(g in s for s in sets_in):
                return False
        # Every non-significant pair must share at least one letter.
        for a, b in ns_pairs:
            if not any((a in s and b in s) for s in sets_in):
                return False
        # No significant pair may share any letter.
        for i, a in enumerate(levels):
            for b in levels[i + 1 :]:
                if sig.get((a, b), False) and any((a in s and b in s) for s in sets_in):
                    return False
        return True

    # Piepho-style split step:
    # start with one common letter, then split whenever a significant pair shares a set.
    letter_sets: list[set[str]] = [set(levels)]
    sig_pairs = [(a, b) for i, a in enumerate(levels) for b in levels[i + 1 :] if sig[(a, b)]]
    for a, b in sig_pairs:
        updated: list[set[str]] = []
        for s in letter_sets:
            if a in s and b in s:
                s1 = set(s)
                s2 = set(s)
                s1.discard(a)
                s2.discard(b)
                if s1:
                    updated.append(s1)
                if s2:
                    updated.append(s2)
            else:
                updated.append(set(s))
        letter_sets = _reduce_sets(updated)

    letter_sets = _reduce_sets(letter_sets)
    if not letter_sets:
        return {g: "" for g in levels}

    # Remove redundant letter columns while preserving CLD constraints.
    changed = True
    while changed and len(letter_sets) > 1:
        changed = False
        for i in range(len(letter_sets) - 1, -1, -1):
            trial = [s for j, s in enumerate(letter_sets) if j != i]
            if _is_valid_cover(trial):
                letter_sets = trial
                changed = True
                break

    rank = {g: i for i, g in enumerate(levels)}  # lower rank => higher mean
    set_indices = list(range(len(letter_sets)))
    top_group = levels[0]

    def _group_positions(order: tuple[int, ...] | list[int]) -> dict[str, list[int]]:
        pos = {old_i: new_i for new_i, old_i in enumerate(order)}
        out: dict[str, list[int]] = {}
        for g in levels:
            idxs = sorted(pos[i] for i, s in enumerate(letter_sets) if g in s)
            out[g] = idxs
        return out

    def _score(order: tuple[int, ...] | list[int]):
        idxs_by_group = _group_positions(order)

        # 1) Force highest-mean group to carry 'a' when possible.
        top_has_a = 0 if (idxs_by_group[top_group] and idxs_by_group[top_group][0] == 0) else 1

        # 2) Highest means should get the earliest letters (a, then b, ... when possible).
        first_pos = tuple((idxs_by_group[g][0] if idxs_by_group[g] else 10**6) for g in levels)

        # 3) Minimize skipped letters inside multi-letter labels (prefer cd over bd).
        gap_penalty = 0
        for g in levels:
            idxs = idxs_by_group[g]
            if len(idxs) >= 2:
                gap_penalty += (idxs[-1] - idxs[0] + 1 - len(idxs))

        # 4) Prefer fewer letters for top-ranked groups (cleaner CLD near top).
        top_complexity = tuple(len(idxs_by_group[g]) for g in levels)

        # 5) Deterministic final tie-break.
        flat = tuple(x for g in levels for x in idxs_by_group[g])
        return top_has_a, gap_penalty, first_pos, top_complexity, flat

    if len(set_indices) <= 8:
        best_order = min(permutations(set_indices), key=_score)
    else:
        # Heuristic for larger sets: prioritize columns touching higher-ranked groups.
        best_order = tuple(
            sorted(
                set_indices,
                key=lambda i: (
                    min(rank[g] for g in letter_sets[i]),
                    len(letter_sets[i]),
                    sorted(rank[g] for g in letter_sets[i]),
                ),
            )
        )

    alphabet = [chr(i) for i in range(ord("a"), ord("z") + 1)]
    mapped_symbol = {}
    for new_i, old_i in enumerate(best_order):
        mapped_symbol[old_i] = alphabet[new_i] if new_i < len(alphabet) else f"a{new_i-len(alphabet)+1}"

    labels: dict[str, str] = {}
    for g in levels:
        chars = [mapped_symbol[i] for i, s in enumerate(letter_sets) if g in s]
        chars = sorted(set(chars), key=lambda c: (len(c) > 1, c))
        labels[g] = "".join(chars)
    for g in levels:
        labels.setdefault(g, "")
    return labels


def pca_biplot_2d(
    df: pd.DataFrame,
    columns: list[str],
    color_col: str | None = None,
    label_col: str | None = None,
):
    use_cols = [c for c in columns if c in df.columns]
    if len(use_cols) < 2:
        return None, None, None, "PCA 至少需要 2 個數值欄位。"

    x = df[use_cols].apply(pd.to_numeric, errors="coerce")
    valid_idx = x.dropna().index
    x = x.loc[valid_idx]
    if len(x) < 3:
        return None, None, None, "PCA 有效樣本不足（需至少 3 筆完整資料）。"

    transform_rows: list[dict[str, str | float]] = []
    x_proc = x.copy()
    for col in x_proc.columns:
        ser = x_proc[col].astype(float)
        q50 = float(ser.quantile(0.5))
        q95 = float(ser.quantile(0.95))
        skew = float(ser.skew())
        long_tail = bool((abs(skew) >= 1.0) and (q50 != 0) and ((q95 / q50) >= 3.0))

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

        transform_rows.append(
            {
                "Variable": col,
                "Skewness": skew,
                "Q95/Q50": (q95 / q50) if q50 != 0 else np.nan,
                "Transform": transform,
                "Note": note,
            }
        )

    std = x_proc.std(ddof=0).replace(0, np.nan)
    z = (x_proc - x_proc.mean()) / std
    z = z.dropna(axis=1)
    if z.shape[1] < 2:
        return None, None, None, "可用於 PCA 的變數不足（可能有常數欄位）。"

    m = z.to_numpy()
    u, s, vt = np.linalg.svd(m, full_matrices=False)
    eigvals = (s**2) / (m.shape[0] - 1)
    var_ratio = eigvals / eigvals.sum()

    scores_all = u * s
    scores = scores_all[:, :2]
    cos2 = (scores[:, 0] ** 2 + scores[:, 1] ** 2) / np.maximum((scores_all**2).sum(axis=1), 1e-12)
    point_size = 11

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
                    marker=dict(size=point_size, line=dict(color="black", width=1), opacity=0.85),
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
                marker=dict(size=point_size, line=dict(color="black", width=1), opacity=0.85, color="white"),
                name="Samples",
            )
        )

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

    # Expand plot ranges so sample labels / vector labels remain inside the frame.
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
        height=560,
        width=860,
    )

    transform_df = pd.DataFrame(transform_rows)

    vector_table = pd.DataFrame(
        {
            "Variable": z.columns,
            "Loading_PC1": loadings[:, 0],
            "Loading_PC2": loadings[:, 1],
            "Vector_X": vectors[:, 0],
            "Vector_Y": vectors[:, 1],
            "Contribution_PC1_%": (loadings[:, 0] ** 2) / np.maximum((loadings[:, 0] ** 2).sum(), 1e-12) * 100,
            "Contribution_PC2_%": (loadings[:, 1] ** 2) / np.maximum((loadings[:, 1] ** 2).sum(), 1e-12) * 100,
        }
    ).sort_values("Contribution_PC1_%", ascending=False)
    vector_table = vector_table.merge(transform_df[["Variable", "Transform", "Note"]], on="Variable", how="left")

    axis_table = pd.DataFrame(
        {
            "Axis": ["Dim1", "Dim2"],
            "Explained_variance_%": [var_ratio[0] * 100, var_ratio[1] * 100],
        }
    )
    return fig, axis_table, vector_table, None


def apply_paper_layout(
    fig: go.Figure,
    title: str,
    x_title: str,
    y_title: str,
    height: int = 560,
    width: int = 900,
) -> go.Figure:
    # A 1.5~1.6 landscape ratio is common for single-panel journal figures.
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


def render_centered_plot(fig: go.Figure):
    left, center, right = st.columns([1, 3, 1])
    with center:
        st.plotly_chart(fig, use_container_width=False)


st.set_page_config(page_title="Agrecolgy Data Analysis Interface", page_icon="assets/app_icon.svg", layout="wide")
st.title("Agrecolgy Data Analysis Interface")

uploaded = st.file_uploader("上傳資料（CSV / XLSX）", type=["csv", "xlsx"])
if not uploaded:
    st.info("請先上傳資料檔。")
    st.stop()

sheet_name = None
if uploaded.name.lower().endswith(".xlsx"):
    xls = pd.ExcelFile(uploaded)
    sheet_name = st.selectbox("選擇工作表", options=xls.sheet_names)
    uploaded.seek(0)

df_raw = load_data(uploaded, sheet_name=sheet_name)
if len(df_raw) < 2:
    st.error("資料列不足：需至少包含第 2 列單位與第 3 列起的資料。")
    st.stop()

# 規格：第 1 列欄名、第 2 列單位、第 3 列起才是資料。
unit_row = df_raw.iloc[0].copy()
df_raw = df_raw.iloc[1:].reset_index(drop=True)
all_cols = df_raw.columns.tolist()

st.subheader("欄位角色設定")
c1, c2 = st.columns(2)
replicate_options = [None] + all_cols
rep_like = next((c for c in all_cols if str(c).lower() == "rep"), None)
rep_default_index = replicate_options.index(rep_like) if rep_like in replicate_options else 0
replicate_col = c1.selectbox("重複欄位（Rep，可選）", options=replicate_options, index=rep_default_index)
param_mode = c2.radio("參數欄位指定方式", options=["從某欄開始", "手動選擇"], horizontal=True)

if param_mode == "從某欄開始":
    default_start = "Dry_matter" if "Dry_matter" in all_cols else all_cols[min(5, len(all_cols) - 1)]
    parameter_start = st.selectbox("參數起始欄位", options=all_cols, index=all_cols.index(default_start))
    parameter_cols = select_parameter_columns(df_raw, start_col=parameter_start, exclude_cols=[replicate_col] if replicate_col else [])
else:
    parameter_cols = st.multiselect("手動選擇參數欄位", options=all_cols)

if not parameter_cols:
    st.error("未找到可分析的參數欄位，請調整『參數起始欄位』或改用手動選擇。")
    st.stop()

df = coerce_numeric_columns(df_raw, parameter_cols)
numeric_cols, categorical_cols = split_columns(df)

# 因子欄位應排除參數欄位
factor_cols = [c for c in all_cols if c not in parameter_cols]
if replicate_col and replicate_col in factor_cols:
    factor_cols = [c for c in factor_cols if c != replicate_col]

for c in factor_cols:
    df[c] = df[c].map(_normalize_factor_label)

st.subheader("資料預覽")
st.dataframe(df, use_container_width=True)
st.write(f"分析參數欄位數：{len(parameter_cols)}")
with st.expander("查看欄位單位（第 2 列）"):
    st.dataframe(pd.DataFrame([unit_row]), use_container_width=True)

st.markdown(
    """
    <style>
    div[data-testid="stPills"] > div {
        overflow-x: auto;
        white-space: nowrap;
        padding-bottom: 0.25rem;
    }
    div[data-testid="stPills"] [data-baseweb="tag"] {
        white-space: nowrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

selected_responses = render_multi_button_selector(
    "選擇反應變數（可複選）",
    options=parameter_cols,
    key_prefix="responses",
    default=parameter_cols[:1],
)

if not selected_responses:
    st.warning("請至少選擇一個反應變數。")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["常態性/前提檢查", "ANOVA", "Post-hoc", "作圖與相關性"])

with tab1:
    group_for_norm = st.selectbox("常態性分組欄位（可選）", options=[None] + factor_cols)
    group_for_levene = st.selectbox("Levene 分組欄位", options=factor_cols)
    qq_group = st.selectbox("QQ Plot 分組欄位（可選）", options=[None] + factor_cols)
    for response in selected_responses:
        st.markdown(f"#### {response}")
        st.dataframe(normality_checks(df, response=response, group=group_for_norm), use_container_width=True)
        st.dataframe(levene_homogeneity(df, response=response, group=group_for_levene), use_container_width=True)
        qq_fig = qqplot_figure(df, response=response, group=qq_group)
        if qq_fig is not None:
            left, center, right = st.columns([1, 2, 1])
            with center:
                st.plotly_chart(qq_fig, use_container_width=True)
        else:
            st.info(f"{response} 的 QQ Plot 樣本數不足（每組至少 3 筆）。")

with tab2:
    st.caption("顯著差異列會以底色標示（p < 0.05）。")
    if replicate_col:
        design_type = st.radio("試驗設計", options=["CRD", "RCBD"], horizontal=True, index=0)
        block_for_anova = replicate_col if design_type == "RCBD" else None
    else:
        design_type = "CRD"
        block_for_anova = None
    st.caption(f"目前設計：{design_type}（{'含 Rep 區集' if block_for_anova else '不含 Rep 區集'}）")

    st.markdown("**ANOVA 子資料集篩選（可選）**")
    f1, f2 = st.columns([2, 3])
    anova_subset_factor = f1.selectbox(
        "篩選欄位",
        options=[None] + [c for c in factor_cols if c != replicate_col],
        key="anova_subset_factor",
    )

    anova_subset_values: list[str] = []
    if anova_subset_factor:
        available_values = sorted(df[anova_subset_factor].dropna().astype(str).unique().tolist())
        default_values = available_values
        if anova_subset_factor == "Fertilizer":
            yes_like = [v for v in available_values if v.strip().upper() == "YES"]
            if yes_like:
                default_values = yes_like
        anova_subset_values = f2.multiselect(
            "保留層級",
            options=available_values,
            default=default_values,
            key="anova_subset_values",
        )
    else:
        f2.empty()

    anova_df = df.copy()
    filter_notes: list[str] = []
    if anova_subset_factor:
        if not anova_subset_values:
            anova_df = anova_df.iloc[0:0]
        else:
            keep_set = {str(v) for v in anova_subset_values}
            anova_df = anova_df[anova_df[anova_subset_factor].astype(str).isin(keep_set)]
            filter_notes.append(f"{anova_subset_factor} in {sorted(keep_set)}")

    st.write(f"ANOVA 使用資料筆數：{len(anova_df)} / {len(df)}")
    if filter_notes:
        st.caption("已套用篩選：" + "；".join(filter_notes))
    if anova_df.empty:
        st.warning("目前篩選條件下沒有可用資料，請調整條件後再執行 ANOVA。")

    with st.form("anova_form"):
        factors = st.multiselect("ANOVA 因子", options=[c for c in factor_cols if c != replicate_col])
        typ = st.selectbox("ANOVA Type", options=[1, 2, 3], index=1)
        run_anova = st.form_submit_button("執行 ANOVA")
    if run_anova:
        if anova_df.empty:
            st.error("子資料集為空，無法執行 ANOVA。")
        else:
            for response in selected_responses:
                st.markdown(f"#### {response}")
                try:
                    anova_result_df = anova_analysis(
                        anova_df,
                        response=response,
                        factors=factors,
                        typ=typ,
                        block_factor=block_for_anova,
                    )
                    st.dataframe(
                        highlight_significant_rows(anova_result_df),
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"{response} ANOVA 執行失敗：{e}")

    with st.form("nested_anova_form"):
        c1, c2, c3 = st.columns(3)
        parent = c1.selectbox("上層因子", options=[c for c in factor_cols if c != replicate_col], key="parent")
        nested = c2.selectbox("巢狀因子", options=[c for c in factor_cols if c != replicate_col], key="nested")
        ntyp = c3.selectbox("ANOVA Type", options=[1, 2, 3], index=1)
        run_nested = st.form_submit_button("執行巢狀 ANOVA")
    if run_nested:
        if parent == nested:
            st.error("上層因子與巢狀因子不可相同。")
        elif anova_df.empty:
            st.error("子資料集為空，無法執行巢狀 ANOVA。")
        else:
            for response in selected_responses:
                st.markdown(f"#### {response}")
                try:
                    nested_result_df = nested_anova(
                        anova_df,
                        response=response,
                        parent_factor=parent,
                        nested_factor=nested,
                        typ=ntyp,
                        block_factor=block_for_anova,
                    )
                    st.dataframe(
                        highlight_significant_rows(nested_result_df),
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"{response} 巢狀 ANOVA 執行失敗：{e}")

with tab3:
    post_group = st.selectbox("Post-hoc 分組欄位", options=[c for c in factor_cols if c != replicate_col])
    method = st.radio("方法", options=["LSD", "Tukey", "Bonferroni", "Dunn"], horizontal=True)
    adjust = None
    if method == "Dunn":
        adjust = st.selectbox("Dunn 校正", options=["bonferroni", "holm", "fdr_bh"])

    if st.button("執行 Post-hoc"):
        for response in selected_responses:
            st.markdown(f"#### {response}")
            if method == "LSD":
                st.dataframe(lsd_posthoc(df, response=response, group=post_group), use_container_width=True)
            elif method == "Tukey":
                st.dataframe(tukey_posthoc(df, response=response, group=post_group), use_container_width=True)
            elif method == "Bonferroni":
                st.dataframe(bonferroni_posthoc(df, response=response, group=post_group), use_container_width=True)
            else:
                st.dataframe(kruskal_wallis(df, response=response, group=post_group), use_container_width=True)
                st.dataframe(dunn_posthoc(df, response=response, group=post_group, p_adjust=adjust), use_container_width=True)

with tab4:
    chart_options = ["散佈圖", "相關性表格", "相關性熱圖", "PCA", "ANOVA 對應圖"]
    selected_charts = render_multi_button_selector(
        "選擇要顯示的內容（可複選）",
        options=chart_options,
        key_prefix="tab4_charts",
        default=chart_options,
    )

    if not selected_charts:
        st.info("請至少選擇一個顯示項目。")
        st.stop()

    st.markdown("**共用資料篩選（適用散佈圖、相關性、PCA，不影響 ANOVA 對應圖）**")
    max_filter_rows = 12
    if "tab4_filter_rows" not in st.session_state:
        st.session_state["tab4_filter_rows"] = 1
    factor_lookup = {str(c): c for c in factor_cols}

    current_rows = int(st.session_state.get("tab4_filter_rows", 1))
    current_rows = max(1, min(current_rows, max_filter_rows))

    # Keep used filters compacted at the top and always reserve exactly one blank row.
    active_filters: list[tuple[object, list[str] | None]] = []
    for i in range(current_rows):
        factor_key = f"tab4_filter_factor_{i}"
        values_key = f"tab4_filter_values_{i}"
        factor_value = st.session_state.get(factor_key)
        factor_name = "" if factor_value is None else str(factor_value)
        if factor_name not in factor_lookup:
            continue

        raw_values = st.session_state.get(values_key)
        if raw_values is None:
            value_list: list[str] | None = None
        elif isinstance(raw_values, list):
            value_list = [str(v) for v in raw_values]
        elif isinstance(raw_values, (tuple, set)):
            value_list = [str(v) for v in raw_values]
        else:
            value_list = [str(raw_values)]
        active_filters.append((factor_lookup[factor_name], value_list))

    desired_rows = min(len(active_filters) + 1, max_filter_rows)

    for i in range(desired_rows):
        factor_key = f"tab4_filter_factor_{i}"
        values_key = f"tab4_filter_values_{i}"
        if i < len(active_filters):
            target_factor, target_values = active_filters[i]
        else:
            target_factor, target_values = None, []

        if st.session_state.get(factor_key) != target_factor:
            st.session_state[factor_key] = target_factor
        if st.session_state.get(values_key) != target_values:
            st.session_state[values_key] = target_values

    for i in range(desired_rows, max_filter_rows):
        factor_key = f"tab4_filter_factor_{i}"
        values_key = f"tab4_filter_values_{i}"
        factor_prev_key = f"tab4_filter_factor_prev_{i}"
        if factor_key in st.session_state:
            st.session_state.pop(factor_key)
        if factor_prev_key in st.session_state:
            st.session_state.pop(factor_prev_key)
        if values_key in st.session_state:
            st.session_state.pop(values_key)

    if st.session_state.get("tab4_filter_rows") != desired_rows:
        st.session_state["tab4_filter_rows"] = desired_rows

    viz_df = df.copy()
    n_filter_rows = desired_rows
    used_any_filter = False
    hierarchy_labels: list[str] = []

    for i in range(n_filter_rows):
        rf1, rf2 = st.columns([2, 3])
        factor_key = f"tab4_filter_factor_{i}"
        values_key = f"tab4_filter_values_{i}"
        factor_prev_key = f"tab4_filter_factor_prev_{i}"

        factor = rf1.selectbox(
            f"第 {i + 1} 層：篩選欄位",
            options=[None] + factor_cols,
            key=factor_key,
        )

        if factor:
            used_any_filter = True
            available_values = _ordered_levels([str(x) for x in viz_df[factor].dropna().tolist()])
            prev_factor = st.session_state.get(factor_prev_key)
            prev_values = st.session_state.get(values_key)
            if prev_factor != factor or prev_values is None:
                st.session_state[values_key] = available_values
            else:
                kept_values = [x for x in prev_values if x in available_values]
                st.session_state[values_key] = kept_values
            st.session_state[factor_prev_key] = factor

            with rf2:
                if hasattr(st, "pills"):
                    selected_values = st.pills(
                        f"第 {i + 1} 層：保留值",
                        options=available_values,
                        selection_mode="multi",
                        key=values_key,
                    )
                else:
                    selected_values = st.multiselect(
                        f"第 {i + 1} 層：保留值",
                        options=available_values,
                        key=values_key,
                    )
            selected_values = selected_values or []
            hierarchy_labels.append(f"第{i + 1}層: {factor}")

            if not selected_values:
                viz_df = viz_df.iloc[0:0]
            else:
                keep_set = set(selected_values)
                viz_df = viz_df[viz_df[factor].astype(str).isin(keep_set)]
        else:
            st.session_state[values_key] = []
            st.session_state[factor_prev_key] = None
            with rf2:
                if hasattr(st, "pills"):
                    st.pills(
                        f"第 {i + 1} 層：保留值",
                        options=[],
                        selection_mode="multi",
                        key=values_key,
                        disabled=True,
                    )
                else:
                    st.multiselect(
                        f"第 {i + 1} 層：保留值",
                        options=[],
                        key=values_key,
                        disabled=True,
                    )

    if used_any_filter:
        st.caption(f"篩選後資料列數：{len(viz_df)} / {len(df)}")
    else:
        st.caption(f"目前資料列數：{len(viz_df)}")

    if hierarchy_labels:
        st.caption("篩選階層：" + " > ".join(hierarchy_labels))
    if "散佈圖" in selected_charts:
        if viz_df.empty:
            st.info("篩選後無資料，無法繪製散佈圖。")
        else:
            c1, c2, c3 = st.columns(3)
            x_col = c1.selectbox("X", options=parameter_cols, key="x_col")
            y_col = c2.selectbox("Y", options=parameter_cols, key="y_col", index=min(1, len(parameter_cols) - 1))
            color_col = c3.selectbox("顏色分組", options=[None] + factor_cols, key="scatter_color_col")
            st.plotly_chart(px.scatter(viz_df, x=x_col, y=y_col, color=color_col, hover_data=all_cols), use_container_width=True)

    if "相關性表格" in selected_charts or "相關性熱圖" in selected_charts:
        corr_method = st.radio("相關係數方法", ["pearson", "spearman"], horizontal=True)
        if viz_df.empty:
            st.info("篩選後無資料，無法計算相關性。")
        else:
            corr = correlation_table(viz_df, method=corr_method, columns=selected_responses)
            if "相關性表格" in selected_charts:
                st.dataframe(corr, use_container_width=True)
            if "相關性熱圖" in selected_charts:
                pmat = _correlation_pvalue_matrix(viz_df, columns=selected_responses, method=corr_method)
                text_mat = pd.DataFrame("", index=corr.index, columns=corr.columns, dtype=object)
                n = len(corr.index)
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            text_mat.iat[i, j] = str(corr.index[i])
                        elif i < j:
                            text_mat.iat[i, j] = _p_to_stars(pmat.iat[i, j])
                        else:
                            v = corr.iat[i, j]
                            text_mat.iat[i, j] = "" if pd.isna(v) else f"{v:.2f}"

                hm = px.imshow(
                    corr,
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                )
                hm.update_traces(text=text_mat.values, texttemplate="%{text}")
                st.plotly_chart(hm, use_container_width=True)

    if "PCA" in selected_charts:
        p1, p2 = st.columns(2)
        pca_color = p1.selectbox("PCA 顏色分組（可選）", options=[None] + factor_cols, key="pca_color_col")
        pca_label = p2.selectbox("PCA 樣本標籤欄位（可選）", options=[None] + all_cols, key="pca_label_col")
        pca_fig, pca_axes, pca_vectors, pca_err = pca_biplot_2d(
            viz_df,
            columns=selected_responses,
            color_col=pca_color,
            label_col=pca_label,
        )
        if pca_err:
            st.info(pca_err)
        else:
            render_centered_plot(pca_fig)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**軸貢獻量 (%)**")
                st.dataframe(pca_axes, use_container_width=True)
            with c2:
                st.markdown("**成分向量與變數貢獻**")
                st.dataframe(pca_vectors, use_container_width=True)
            if "Transform" in pca_vectors.columns:
                log_cols = pca_vectors.loc[pca_vectors["Transform"] == "log10_then_standardize", "Variable"].tolist()
                if log_cols:
                    st.caption("PCA 前處理：已對長尾欄位先做 log10(X) 再標準化 -> " + ", ".join(map(str, log_cols)))
                else:
                    st.caption("PCA 前處理：所有欄位皆為標準化（未觸發長尾 log10 轉換）。")

    if "ANOVA 對應圖" in selected_charts:
        st.markdown("#### ANOVA 對應圖（依 tab2 設定自動判斷）")
        if anova_df.empty:
            st.info("ANOVA 子資料集為空，無法產生對應圖。")
        elif not factors:
            st.info("請先在 ANOVA 分頁選擇至少一個因子。")
        else:
            for response in selected_responses:
                st.markdown(f"##### {response}")
                try:
                    anova_tbl = anova_analysis(
                        anova_df,
                        response=response,
                        factors=factors,
                        typ=typ,
                        block_factor=block_for_anova,
                    )
                except Exception as e:
                    st.error(f"{response} ANOVA 計算失敗：{e}")
                    continue

                p_col = _anova_p_col(anova_tbl)
                if p_col is None:
                    st.info("找不到 ANOVA p-value 欄位，無法判斷對應圖型。")
                    continue

                sig = anova_tbl.copy()
                sig[p_col] = pd.to_numeric(sig[p_col], errors="coerce")
                sig = sig[
                    sig[p_col].notna()
                    & (sig[p_col] < 0.05)
                    & (sig["term"].astype(str).str.lower() != "residual")
                ]

                inter_terms = [t for t in sig["term"].astype(str).tolist() if ":" in t]
                main_terms = [t for t in sig["term"].astype(str).tolist() if ":" not in t]

                # 1) Draw interaction plots for all significant 2-way terms.
                drawn_interactions: set[tuple[str, str]] = set()
                for term in inter_terms:
                    response_y_label_mean = _format_response_y_label(response, unit_row=unit_row, with_mean=True)
                    facs = _extract_factor_names_from_term(term)
                    if len(facs) != 2:
                        st.info(f"偵測到 `{term}`，但目前只自動繪製二因子 interaction plot。")
                        continue
                    f1, f2 = facs
                    pair = tuple(sorted((f1, f2)))
                    if pair in drawn_interactions:
                        continue
                    if not all(f in anova_df.columns for f in (f1, f2)):
                        st.info(f"顯著交互作用 `{term}` 因子欄位不存在，略過。")
                        continue

                    tmp = anova_df[[response, f1, f2]].copy()
                    tmp[response] = pd.to_numeric(tmp[response], errors="coerce")
                    tmp = tmp.dropna()
                    if tmp.empty:
                        st.info(f"{f1} × {f2} 互作圖資料不足。")
                        continue

                    means = tmp.groupby([f1, f2], observed=False)[response].mean().reset_index()
                    x_order = _ordered_levels(tmp[f1].astype(str).dropna().tolist())
                    color_order = _ordered_levels(tmp[f2].astype(str).dropna().tolist())
                    means[f1] = means[f1].astype(str)
                    means[f2] = means[f2].astype(str)
                    fig = px.line(
                        means,
                        x=f1,
                        y=response,
                        color=f2,
                        markers=True,
                        category_orders={f1: x_order, f2: color_order},
                    )
                    fig = apply_paper_layout(
                        fig,
                        title=f"{response} Interaction: {f1} × {f2}",
                        x_title=f1,
                        y_title=response_y_label_mean,
                        height=560,
                        width=860,
                    )
                    fig.update_traces(line=dict(width=2), marker=dict(size=8))
                    render_centered_plot(fig)
                    st.caption(f"依顯著交互作用 `{term}` 產生 interaction plot。")
                    drawn_interactions.add(pair)

                # 2) Draw bar charts for all significant main effects.
                main_factors: list[str] = []
                for term in main_terms:
                    facs = _extract_factor_names_from_term(term)
                    if facs and facs[0] in anova_df.columns:
                        main_factors.append(facs[0])
                if not main_factors and factors:
                    # If nothing significant, keep one fallback effect for quick visualization.
                    if factors[0] in anova_df.columns:
                        main_factors = [factors[0]]
                main_factors = list(dict.fromkeys(main_factors))

                if not main_factors and not inter_terms:
                    st.info("未偵測到可繪圖的顯著主效應/交互作用。")

                for effect_factor in main_factors:
                    response_y_label = _format_response_y_label(response, unit_row=unit_row, with_mean=False)
                    tmp = anova_df[[response, effect_factor]].copy()
                    tmp[effect_factor] = tmp[effect_factor].map(_normalize_factor_label)
                    tmp[response] = pd.to_numeric(tmp[response], errors="coerce")
                    tmp = tmp.dropna()
                    if tmp.empty:
                        st.info(f"{effect_factor} 柱狀圖資料不足。")
                        continue

                    summary = tmp.groupby(effect_factor, observed=False)[response].agg(mean="mean", sd="std", n="count").reset_index()
                    summary[effect_factor] = summary[effect_factor].map(_normalize_factor_label)
                    summary["sd"] = summary["sd"].fillna(0.0)
                    level_order = _ordered_levels(tmp[effect_factor].dropna().tolist())
                    summary[effect_factor] = pd.Categorical(summary[effect_factor], categories=level_order, ordered=True)
                    summary = summary.sort_values(effect_factor).copy()
                    summary[effect_factor] = summary[effect_factor].astype(str)

                    try:
                        cld_order = summary.sort_values("mean", ascending=False)[effect_factor].tolist()
                        dunn_adjust = adjust if (method == "Dunn" and adjust) else "bonferroni"
                        sig_map = _pairwise_significance_for_cld(
                            tmp,
                            response=response,
                            group=effect_factor,
                            method=method,
                            dunn_adjust=dunn_adjust,
                        )
                        cld_map = _make_cld_from_significance(sig_map, group_order=cld_order)
                    except Exception:
                        cld_map = {}
                    summary["CLD"] = summary[effect_factor].map(cld_map).fillna("")

                    bar = go.Figure()
                    bar.add_trace(
                        go.Bar(
                            x=summary[effect_factor],
                            y=summary["mean"],
                            error_y=dict(type="data", array=summary["sd"], thickness=1.4, width=4),
                            marker_line=dict(width=0.8, color="black"),
                            showlegend=False,
                            name="",
                        )
                    )
                    y_top = summary["mean"] + summary["sd"]
                    y_bottom = summary["mean"] - summary["sd"]
                    data_span = float(y_top.max() - y_bottom.min())
                    data_span = max(data_span, float(np.abs(y_top.max())) * 0.2, 1e-9)
                    cld_text = summary["CLD"].astype(str).str.strip()
                    has_label_mask = cld_text.ne("")
                    label_offset = data_span * 0.08
                    label_y = (y_top + label_offset).astype(float)
                    has_cld = bool(has_label_mask.any())
                    if has_cld:
                        bar.add_trace(
                            go.Scatter(
                                x=summary.loc[has_label_mask, effect_factor],
                                y=label_y.loc[has_label_mask],
                                mode="text",
                                text=cld_text.loc[has_label_mask],
                                textposition="top center",
                                textfont=dict(size=15, color="black"),
                                showlegend=False,
                                hoverinfo="skip",
                                cliponaxis=False,
                            )
                        )

                    y_min = float(y_bottom.min())
                    y_max = float(y_top.max())
                    cld_y_max = float(label_y.loc[has_label_mask].max()) if has_cld else y_max
                    bar = apply_paper_layout(
                        bar,
                        title=f"{response} by {effect_factor} (mean ± SD, {method})",
                        x_title=effect_factor,
                        y_title=response_y_label,
                        height=560,
                        width=860,
                    )
                    bar.update_xaxes(categoryorder="array", categoryarray=level_order, type="category")
                    lower_pad = data_span * 0.08
                    upper_pad = data_span * (0.12 if has_cld else 0.06)
                    if y_min >= 0:
                        y_axis_min = max(0.0, y_min - lower_pad)
                    else:
                        y_axis_min = y_min - lower_pad
                    y_axis_max = max(y_max, cld_y_max) + upper_pad
                    bar.update_yaxes(range=[y_axis_min, y_axis_max])
                    render_centered_plot(bar)
                    st.dataframe(summary, use_container_width=True)

st.markdown("---")
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="cleaned_data", index=False)
st.download_button(
    "下載清理後資料（XLSX）",
    data=buffer.getvalue(),
    file_name="cleaned_data.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
