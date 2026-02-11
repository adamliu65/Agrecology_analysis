"""
Utilities for Compact Letter Display (CLD) generation from statistical tests.
"""

from __future__ import annotations

import re
from itertools import permutations
from typing import Optional

import pandas as pd

from .constants import DEFAULT_ALPHA


def _extract_factor_names_from_term(term: str) -> list[str]:
    """Extract factor names from ANOVA term (e.g., 'C(factor1)' -> 'factor1')."""
    return re.findall(r"C\(([^)]+)\)", str(term))


def _ordered_levels(values: list[str]) -> list[str]:
    """
    Order factor levels intelligently.
    
    Prefers natural ordering (T1, T2, ...) when pattern is consistent,
    falls back to alphabetic ordering.
    
    Parameters
    ----------
    values : list[str]
        List of factor level values.
    
    Returns
    -------
    list[str]
        Ordered levels.
    """
    vals = [str(v) for v in values]
    uniq = list(dict.fromkeys(vals))
    if not uniq:
        return uniq

    # Try to match T1, T2, ... T8 pattern
    m = [re.match(r"^([A-Za-z]+)(\d+)$", v) for v in uniq]
    if all(x is not None for x in m):
        prefixes = {x.group(1) for x in m if x is not None}
        if len(prefixes) == 1:
            return sorted(uniq, key=lambda s: int(re.match(r"^[A-Za-z]+(\d+)$", s).group(1)))

    if all(re.match(r"^T\d+$", v) for v in uniq):
        return sorted(uniq, key=lambda s: int(s[1:]))

    return sorted(uniq)


def pairwise_significance_for_cld(
    df: pd.DataFrame,
    response: str,
    group: str,
    method: str,
    posthoc_func_map: dict,
    dunn_adjust: str = "bonferroni",
) -> dict[tuple[str, str], bool]:
    """
    Build significance matrix from post-hoc tests.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    response : str
        Numeric response column name.
    group : str
        Grouping column name.
    method : str
        Post-hoc method ("LSD", "Tukey", "Bonferroni", or "Dunn").
    posthoc_func_map : dict
        Dictionary mapping method names to functions:
        {"LSD": lsd_func, "Tukey": tukey_func, ...}
    dunn_adjust : str, default="bonferroni"
        Adjustment method for Dunn test.
    
    Returns
    -------
    dict[tuple[str, str], bool]
        Significance map: {(group_a, group_b): is_significant, ...}
    """
    levels = sorted(df[group].dropna().astype(str).unique().tolist())
    sig = {(a, b): False for a in levels for b in levels}

    try:
        if method == "LSD":
            ph = posthoc_func_map["lsd_posthoc"](df, response=response, group=group)
            for _, row in ph.iterrows():
                a = str(row.get("group_a"))
                b = str(row.get("group_b"))
                rej = bool(row.get("significant_at_0.05", False))
                if a in levels and b in levels:
                    sig[(a, b)] = rej
                    sig[(b, a)] = rej
        elif method == "Tukey":
            ph = posthoc_func_map["tukey_posthoc"](df, response=response, group=group)
            for _, row in ph.iterrows():
                a = str(row.get("group_a"))
                b = str(row.get("group_b"))
                rej = row.get("reject_at_0.05", False)
                if isinstance(rej, str):
                    rej = rej.strip().lower() == "true"
                if a in levels and b in levels:
                    sig[(a, b)] = bool(rej)
                    sig[(b, a)] = bool(rej)
        elif method == "Bonferroni":
            ph = posthoc_func_map["bonferroni_posthoc"](df, response=response, group=group)
            for _, row in ph.iterrows():
                a = str(row.get("group_a"))
                b = str(row.get("group_b"))
                rej = bool(row.get("significant_at_0.05", False))
                if a in levels and b in levels:
                    sig[(a, b)] = rej
                    sig[(b, a)] = rej
        elif method == "Dunn":
            ph = posthoc_func_map["dunn_posthoc"](
                df, response=response, group=group, p_adjust=dunn_adjust
            )
            ph = ph.copy()
            if "group" in ph.columns:
                ph["group"] = ph["group"].astype(str)
                for _, row in ph.iterrows():
                    a = str(row["group"])
                    for b in levels:
                        if b not in ph.columns:
                            continue
                        import pandas as pd
                        p = pd.to_numeric(row.get(b), errors="coerce")
                        if pd.notna(p):
                            rej = bool(p < DEFAULT_ALPHA)
                            sig[(a, b)] = rej
                            sig[(b, a)] = rej
    except Exception:
        # Keep conservative default (no significant differences) if post-hoc fails
        pass

    return sig


def make_cld_from_significance(
    sig: dict[tuple[str, str], bool],
    group_order: list[str]
) -> dict[str, str]:
    """
    Generate Compact Letter Display (CLD) from significance matrix.
    
    Uses Piepho's algorithm to split groups based on significant differences,
    then assigns letters optimally.
    
    Parameters
    ----------
    sig : dict[tuple[str, str], bool]
        Significance map from pairwise tests.
    group_order : list[str]
        Order of groups (typically by mean, descending).
    
    Returns
    -------
    dict[str, str]
        Mapping of group -> CLD letters (e.g., {"A": "a", "B": "ab", "C": "b"})
    """
    levels = [str(x) for x in group_order]
    if not levels:
        return {}

    def _reduce_sets(sets_in: list[set[str]]) -> list[set[str]]:
        """Remove duplicate and redundant sets."""
        unique = []
        seen = set()
        for s in sets_in:
            fs = frozenset(s)
            if fs and fs not in seen:
                seen.add(fs)
                unique.append(set(s))

        # Remove strict subsets
        out = []
        for i, s in enumerate(unique):
            if any(i != j and s < t for j, t in enumerate(unique)):
                continue
            out.append(s)
        return out

    ns_pairs = [
        (a, b) for i, a in enumerate(levels) for b in levels[i + 1 :]
        if not sig.get((a, b), False)
    ]

    def _is_valid_cover(sets_in: list[set[str]]) -> bool:
        """Check if letter assignment is valid (satisfies all constraints)."""
        if not sets_in:
            return False
        # Every group must have at least one letter
        for g in levels:
            if not any(g in s for s in sets_in):
                return False
        # Non-significant pairs must share at least one letter
        for a, b in ns_pairs:
            if not any((a in s and b in s) for s in sets_in):
                return False
        # Significant pairs must NOT share any letter
        for i, a in enumerate(levels):
            for b in levels[i + 1 :]:
                if sig.get((a, b), False) and any((a in s and b in s) for s in sets_in):
                    return False
        return True

    # Start with all groups sharing one letter, then split significant pairs
    letter_sets: list[set[str]] = [set(levels)]
    sig_pairs = [
        (a, b) for i, a in enumerate(levels) for b in levels[i + 1 :]
        if sig[(a, b)]
    ]
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

    # Remove redundant letter columns while preserving constraints
    changed = True
    while changed and len(letter_sets) > 1:
        changed = False
        for i in range(len(letter_sets) - 1, -1, -1):
            trial = [s for j, s in enumerate(letter_sets) if j != i]
            if _is_valid_cover(trial):
                letter_sets = trial
                changed = True
                break

    rank = {g: i for i, g in enumerate(levels)}

    def _group_positions(order: tuple[int, ...] | list[int]) -> dict[str, list[int]]:
        """Map groups to their letter column indices."""
        pos = {old_i: new_i for new_i, old_i in enumerate(order)}
        out: dict[str, list[int]] = {}
        for g in levels:
            idxs = sorted(pos[i] for i, s in enumerate(letter_sets) if g in s)
            out[g] = idxs
        return out

    def _score(order: tuple[int, ...] | list[int]):
        """Score a letter column ordering."""
        idxs_by_group = _group_positions(order)

        # Prefer top-ranked group to carry 'a'
        top_group = levels[0]
        top_has_a = 0 if (idxs_by_group[top_group] and idxs_by_group[top_group][0] == 0) else 1

        # Prefer early letters for high-ranked groups
        first_pos = tuple(
            (idxs_by_group[g][0] if idxs_by_group[g] else 10**6) for g in levels
        )

        # Minimize gaps (prefer 'cd' over 'bd')
        gap_penalty = 0
        for g in levels:
            idxs = idxs_by_group[g]
            if len(idxs) >= 2:
                gap_penalty += (idxs[-1] - idxs[0] + 1 - len(idxs))

        # Prefer fewer letters for high-ranked groups
        top_complexity = tuple(len(idxs_by_group[g]) for g in levels)

        # Deterministic tie-break
        flat = tuple(x for g in levels for x in idxs_by_group[g])

        return top_has_a, gap_penalty, first_pos, top_complexity, flat

    # Find optimal letter order
    set_indices = list(range(len(letter_sets)))
    if len(set_indices) <= 8:
        best_order = min(permutations(set_indices), key=_score)
    else:
        # Heuristic for large sets
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

    # Assign letters
    alphabet = [chr(i) for i in range(ord("a"), ord("z") + 1)]
    mapped_symbol = {}
    for new_i, old_i in enumerate(best_order):
        mapped_symbol[old_i] = (
            alphabet[new_i] if new_i < len(alphabet) else f"a{new_i - len(alphabet) + 1}"
        )

    # Build final CLD
    labels: dict[str, str] = {}
    for g in levels:
        chars = [mapped_symbol[i] for i, s in enumerate(letter_sets) if g in s]
        chars = sorted(set(chars), key=lambda c: (len(c) > 1, c))
        labels[g] = "".join(chars)
    for g in levels:
        labels.setdefault(g, "")
    return labels
