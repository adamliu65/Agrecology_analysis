"""
Agrecology Analysis Tool - Statistical analysis toolkit for agronomy data.

This package provides both CLI and Streamlit UI interfaces for comprehensive
statistical analysis including ANOVA, post-hoc tests, normality checks, and
visualization (PCA, correlation heatmaps, effect plots).
"""

from .constants import *
from .data_loader import (
    coerce_numeric_columns,
    load_data,
    load_data_from_path,
    sanitize_columns,
    select_parameter_columns,
    split_columns,
)
from .posthoc_tests import (
    bonferroni_posthoc,
    dunn_posthoc,
    lsd_posthoc,
    tukey_posthoc,
)
from .statistical_analysis import (
    anova_analysis,
    correlation_table,
    kruskal_wallis,
    levene_homogeneity,
    nested_anova,
    normality_checks,
)
from .visualization import (
    apply_paper_layout,
    pca_biplot_2d,
    qqplot_figure,
)

__version__ = "1.0.0"
__author__ = "Agrecology Team"

__all__ = [
    # Data loading
    "load_data",
    "load_data_from_path",
    "sanitize_columns",
    "split_columns",
    "select_parameter_columns",
    "coerce_numeric_columns",
    # Statistical analysis
    "normality_checks",
    "levene_homogeneity",
    "anova_analysis",
    "nested_anova",
    "kruskal_wallis",
    "correlation_table",
    # Post-hoc tests
    "lsd_posthoc",
    "bonferroni_posthoc",
    "tukey_posthoc",
    "dunn_posthoc",
    # Visualization
    "qqplot_figure",
    "pca_biplot_2d",
    "apply_paper_layout",
]
