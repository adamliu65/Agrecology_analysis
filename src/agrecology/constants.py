"""
Global constants and magic numbers for Agrecology Analysis Tool.
"""

# Statistical Analysis Constants
DEFAULT_ALPHA = 0.05
DEFAULT_ANOVA_TYPE = 2  # Type II sum of squares
DEFAULT_MIN_NUMERIC_RATIO = 0.6
MIN_SAMPLES_FOR_SHAPIRO = 3
MIN_GROUPS_FOR_LEVENE = 2

# Dunn Post-Hoc Adjustment Methods
DUNN_ADJUST_METHODS = ["bonferroni", "holm", "fdr_bh"]
DEFAULT_DUNN_ADJUST = "bonferroni"

# PCA Constants
MIN_SAMPLES_FOR_PCA = 3
MIN_VARS_FOR_PCA = 2
PCA_SKEW_THRESHOLD = 1.0
PCA_RATIO_THRESHOLD = 3.0
PCA_POINT_SIZE = 11

# Streamlit UI Constants
PCA_BIPLOT_HEIGHT = 700
QQPLOT_HEIGHT = 520
CORRELATION_HEATMAP_HEIGHT = 600
EFFECT_PLOT_HEIGHT = 500
EFFECT_PLOT_WIDTH = 800

# Column Sanitization
COL_REPLACE_MAP = {
    " ": "_",
    "-": "_",
    "(": "",
    ")": "",
    ":": "_",
    "/": "_",
    "*": "",
}

# Excel/CSV Handling
ASSUMED_UNITS_ROW = 0  # First data row is treated as units row in UI
