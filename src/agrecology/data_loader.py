"""
Data loading and preprocessing utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .constants import COL_REPLACE_MAP, DEFAULT_MIN_NUMERIC_RATIO


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names for formula processing.
    
    Replaces spaces, hyphens, and special characters with underscores
    to make column names safe for statistical formulas.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with potentially problematic column names.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with sanitized column names.
    """
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
            .replace(".", "_")
            .replace("*", "")
        )
        for col in df.columns
    }
    return df.rename(columns=rename_map)


def load_data(uploaded_file: Any, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """
    Load data from CSV or Excel file (Streamlit uploaded file object).
    
    Parameters
    ----------
    uploaded_file : Any
        Streamlit UploadedFile object.
    sheet_name : Optional[str]
        Sheet name for Excel files. If None, uses first sheet.
    
    Returns
    -------
    pd.DataFrame
        Loaded and sanitized DataFrame.
    
    Raises
    ------
    ValueError
        If file format is not CSV or XLSX.
    """
    filename = uploaded_file.name.lower()
    
    if filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
    else:
        raise ValueError(f"Unsupported file format: {filename}")
    
    return sanitize_columns(df)


def load_data_from_path(
    path: str | Path,
    sheet_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Load data from file path (for CLI usage).
    
    Parameters
    ----------
    path : str | Path
        File path to CSV or Excel file.
    sheet_name : Optional[str]
        Sheet name for Excel files. If None, uses first sheet.
    
    Returns
    -------
    pd.DataFrame
        Loaded and sanitized DataFrame.
    
    Raises
    ------
    ValueError
        If file format is not CSV or XLSX.
    """
    p = Path(path)
    
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    elif p.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(p, sheet_name=sheet_name)
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}")
    
    return sanitize_columns(df)


def split_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Split DataFrame columns into numeric and categorical columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    
    Returns
    -------
    tuple[list[str], list[str]]
        (numeric_columns, categorical_columns)
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def select_parameter_columns(
    df: pd.DataFrame,
    start_col: Optional[str] = None,
    manual_cols: Optional[list[str]] = None,
    exclude_cols: Optional[list[str]] = None,
    min_numeric_ratio: float = DEFAULT_MIN_NUMERIC_RATIO,
) -> list[str]:
    """
    Select numeric parameter columns from DataFrame.
    
    Can auto-select from a start column or use manual selection.
    Filters columns by numeric data ratio.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    start_col : Optional[str]
        Column to start from for auto-selection. If None, starts from first column.
    manual_cols : Optional[list[str]]
        Explicitly selected columns. Takes precedence over auto-selection.
    exclude_cols : Optional[list[str]]
        Columns to exclude from selection.
    min_numeric_ratio : float, default=0.6
        Minimum ratio of non-null numeric values required for a column.
    
    Returns
    -------
    list[str]
        Selected parameter column names.
    """
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
    """
    Coerce specified columns to numeric type.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list[str]
        Column names to coerce to numeric.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with coerced columns (NaN for non-convertible values).
    """
    out = df.copy()
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out
