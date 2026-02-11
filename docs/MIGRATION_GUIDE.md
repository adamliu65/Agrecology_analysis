# Project Restructuring Guide

This document explains the reorganization of the Agrecology Analysis Tool for improved maintainability.

## What Changed

### Before (Monolithic)
```
project/
├── app.py                    # 1088 lines - UI + data + stats
├── analysis_pipeline.py      # 334 lines - mixed concerns
├── pipeline_cli.py          # Small CLI wrapper
└── requirements.txt
```

### After (Modular)
```
project/
├── src/agrecology/          # Importable package
│   ├── __init__.py
│   ├── constants.py         # 📌 Magic numbers & config
│   ├── data_loader.py       # 📊 File I/O & preprocessing
│   ├── statistical_analysis.py  # 📈 ANOVA, tests
│   ├── posthoc_tests.py     # 🔬 Pairwise comparisons
│   ├── visualization.py     # 📉 Plotly figures
│   ├── cld_utils.py        # 🔤 Letter display algorithm
│   └── cli.py              # ⌨️ CLI entry point
├── config/                  # Config files (future)
├── tests/                   # Unit tests (future)
├── docs/                    # Documentation
├── app.py                   # Streamlit UI (refactored)
├── pipeline_cli.py          # Legacy CLI wrapper
├── pyproject.toml          # Modern packaging
└── requirements.txt        # Pinned versions
```

## Benefits

| Before | After |
|--------|-------|
| ❌ 1088-line monolith hard to test | ✅ Focused modules, easy to test |
| ❌ Magic numbers scattered | ✅ `constants.py` for easy tuning |
| ❌ Tight coupling | ✅ Loose coupling, import what you need |
| ❌ Floating versions `>=1.36` | ✅ Pinned versions `==1.40.1` |
| ❌ Unclear API | ✅ Documented public API in `__init__.py` |
| ❌ Hard to reuse code | ✅ Can import functions for custom workflows |

## Migration Steps for Existing Code

### Step 1: Update Imports (Old → New)

**Old approach (don't use):**
```python
from analysis_pipeline import anova_analysis, normality_checks
```

**New approach (use this):**
```python
from src.agrecology import anova_analysis, normality_checks
# OR (if installed as package)
from agrecology import anova_analysis, normality_checks
```

### Step 2: Run Tests

Ensure your existing code still works:
```bash
# Install as editable package
pip install -e .

# Or just add src/ to PYTHONPATH
$env:PYTHONPATH = "$env:PYTHONPATH;.\src"

# Run your scripts
python your_script.py
```

### Step 3: Update Configuration

Instead of changing function defaults, edit `constants.py`:

**Before (don't do this):**
```python
def anova_analysis(..., typ: int = 2, ...):  # Hard to change
```

**After (do this):**
```python
# In src/agrecology/constants.py
DEFAULT_ANOVA_TYPE = 2

# In statistical_analysis.py
def anova_analysis(..., typ: int = DEFAULT_ANOVA_TYPE, ...):
```

Then change it project-wide:
```python
# src/agrecology/constants.py
DEFAULT_ANOVA_TYPE = 3  # Change once, affects everywhere
```

## Key Files Reference

### `constants.py`
All magic numbers and configuration:
```python
DEFAULT_ALPHA = 0.05
DEFAULT_ANOVA_TYPE = 2
PCA_SKEW_THRESHOLD = 1.0
QQPLOT_HEIGHT = 520
# ... etc
```

### `data_loader.py`
File I/O and data preprocessing:
```python
load_data(uploaded_file)
load_data_from_path(path)
sanitize_columns(df)
split_columns(df)
select_parameter_columns(df, ...)
coerce_numeric_columns(df, cols)
```

### `statistical_analysis.py`
Core statistical tests:
```python
normality_checks(df, response, group=None)
levene_homogeneity(df, response, group)
anova_analysis(df, response, factors, typ, block_factor)
nested_anova(df, response, parent_factor, nested_factor, ...)
kruskal_wallis(df, response, group)
correlation_table(df, method, columns)
```

### `posthoc_tests.py`
Pairwise comparison methods:
```python
lsd_posthoc(df, response, group)
bonferroni_posthoc(df, response, group)
tukey_posthoc(df, response, group)
dunn_posthoc(df, response, group, p_adjust)
```

### `visualization.py`
Publication-ready Plotly figures:
```python
qqplot_figure(df, response, group=None)
pca_biplot_2d(df, columns, color_col, label_col)
apply_paper_layout(fig, title, x_title, y_title, height, width)
```

### `cld_utils.py`
Compact Letter Display generation:
```python
pairwise_significance_for_cld(df, response, group, method, ...)
make_cld_from_significance(sig_map, group_order)
_ordered_levels(values)  # Smart ordering
```

## Backward Compatibility

The old imports still work (mostly):
```python
# ⚠️ Old approach (still works, but not recommended)
from analysis_pipeline import anova_analysis
```

But we recommend migrating to the new structure:
```python
# ✅ New approach (recommended)
from agrecology import anova_analysis
```

## Installing as Package

```bash
# Development install (recommended for now)
pip install -e .

# Production install (when released to PyPI)
pip install agrecology-analysis
```

This makes the package importable from anywhere:
```python
from agrecology import load_data, anova_analysis
```

## Next Steps

1. ✅ **Review** the new structure
2. ✅ **Test** that your code still works
3. 🔜 **Update** imports in your scripts
4. 🔜 **Extend** with custom functionality using the modular API
5. 🔜 **Contribute** improvements back

## Questions?

See [README.md](README.md) for API documentation and examples.
