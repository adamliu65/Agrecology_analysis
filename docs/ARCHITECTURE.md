# Architecture & Design Decisions

## Overview

The Agrecology Analysis Tool follows **layered architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────┐
│        User Interfaces (app.py)         │
│  (Streamlit UI + pipeline_cli.py)       │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│     Business Logic Layer                │
│  (ANOVA, tests, visualizations)         │
├──────────────────────────────────────────┤
│  • statistical_analysis.py              │
│  • posthoc_tests.py                     │
│  • visualization.py                     │
│  • cld_utils.py                         │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│     Data Layer                          │
│  (Load, clean, preprocess)              │
├──────────────────────────────────────────┤
│  • data_loader.py                       │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│     Configuration                       │
│  (Magic numbers, defaults)              │
├──────────────────────────────────────────┤
│  • constants.py                         │
└─────────────────────────────────────────┘
```

## Module Responsibilities

### `constants.py` - Configuration Hub
**Purpose:** Single source of truth for all magic numbers and defaults.

**Why:**
- Changing `DEFAULT_ALPHA` affects all functions automatically
- No scattered `0.05` values across codebase
- Easy for users to customize behavior project-wide
- Clear what values are configurable

**Example:**
```python
# Change significance level everywhere
DEFAULT_ALPHA = 0.01  # More stringent
```

### `data_loader.py` - ETL Layer
**Purpose:** Extract, Transform, Load data files.

**Responsibilities:**
- CSV/XLSX file I/O
- Column name sanitization (spaces → underscores)
- Data type coercion
- Column classification (numeric vs categorical)
- Parameter column selection

**Design:**
- Two variants: `load_data()` for Streamlit, `load_data_from_path()` for CLI
- `select_parameter_columns()` supports auto-selection or manual specification
- Validates data integrity before returning

### `statistical_analysis.py` - Core Analytics
**Purpose:** Hypothesis testing and ANOVA computations.

**Functions:**
| Function | Purpose |
|----------|---------|
| `normality_checks()` | Shapiro-Wilk test |
| `levene_homogeneity()` | Variance homogeneity |
| `anova_analysis()` | Factorial/blocked ANOVA |
| `nested_anova()` | Nested factor designs |
| `kruskal_wallis()` | Non-parametric ANOVA |
| `correlation_table()` | Pearson/Spearman |

**Design Patterns:**
- **Defensive cleaning**: `_clean_factor_list()` removes confounded factors
- **Graceful fallback**: `_safe_anova_table()` tries Type II, then Type I, then robust
- **Assumption checks**: Validates min group sizes, non-null levels

### `posthoc_tests.py` - Pairwise Comparisons
**Purpose:** Methods for comparing group means after ANOVA.

**Methods:**
| Test | Best For | Correction |
|------|----------|-----------|
| LSD | Quick pairwise | None |
| Tukey HSD | Balanced designs | Automatic |
| Bonferroni | Conservative | Family-wise |
| Dunn | Non-parametric | Flexible |

**Design:**
- Each function returns DataFrame with standardized columns
- CLD compatible: includes pairwise significance information

### `visualization.py` - Publication Plots
**Purpose:** Create journal-quality figures with Plotly.

**Features:**
| Function | Output |
|----------|--------|
| `qqplot_figure()` | Q-Q plot for normality |
| `pca_biplot_2d()` | PCA with vectors + auto log-transform |
| `apply_paper_layout()` | Consistent styling |

**Design:**
- PCA automatically detects long-tailed distributions and log-transforms
- All figures use `apply_paper_layout()` for consistency
- Returns figure + metadata tables

### `cld_utils.py` - Compact Letter Display
**Purpose:** Algorithm for letter groupings from significance matrix.

**Algorithm:**
1. **Piepho's split**: Start with all groups in one letter set
2. **Split significant pairs**: Separate groups that differ
3. **Optimize letters**: Assign a, b, c, ... to minimize label clutter
4. **Smart ordering**: Top-ranked groups get 'a', prefer early letters

**Design:**
- Handles complex significance patterns
- Deterministic (same input always same output)
- Prefers intuitive orderings (T1, T2, not T8, T1)

## Data Flow

### Streamlit UI (`app.py`)
```
1. User uploads file
   ↓
2. load_data() → sanitize + split columns
   ↓
3. User selects parameters, factors, response
   ↓
4. Assumption tests (normality_checks, levene_homogeneity)
   ↓
5. ANOVA (anova_analysis with optional block_factor)
   ↓
6. Post-hoc (lsd_posthoc, tukey_posthoc, dunn_posthoc)
   ↓
7. CLD (make_cld_from_significance)
   ↓
8. Visualization (pca_biplot_2d, qqplot_figure, effect plots)
   ↓
9. Export cleaned data as XLSX
```

### CLI (`pipeline_cli.py`)
```
1. Parse command-line arguments
   ↓
2. load_data_from_path() → coerce numeric columns
   ↓
3. select_parameter_columns() by name/range
   ↓
4. Run all tests → write CSV files
   ↓
5. Report completion with file list
```

## Design Principles

### 1. **Single Responsibility**
Each module has one clear job:
- Load data → `data_loader.py`
- Test hypotheses → `statistical_analysis.py`
- Compare groups → `posthoc_tests.py`
- Draw plots → `visualization.py`

### 2. **Dependency Injection**
Functions receive data as parameters, not global state:
```python
# ❌ Bad: Global state
ALPHA = 0.05
def test(...):
    if p < ALPHA: ...

# ✅ Good: Parameter
def test(..., alpha=0.05):
    if p < alpha: ...
```

### 3. **Defensive Programming**
- Check factor levels, non-null values
- Graceful fallbacks for edge cases
- Meaningful error messages

### 4. **Composition over Inheritance**
- Functions compose into workflows
- No class hierarchies
- Easy to reorder steps

### 5. **Configuration Centralization**
- All defaults in `constants.py`
- No magic numbers in function bodies
- User-modifiable without code edits

## Testing Strategy

### Unit Tests (tests/)
Future: Test each function in isolation
```python
def test_normality_checks_all_groups():
    df = pd.DataFrame({'response': [1,2,3,4,5], ...})
    result = normality_checks(df, response='response')
    assert result.shape[0] == 1
    assert 'p_value' in result.columns
```

### Integration Tests
Future: Test workflows end-to-end
```python
def test_anova_workflow():
    df = load_sample_data()
    anova_result = anova_analysis(df, response='Yield', factors=['Treatment'])
    assert not anova_result.empty
```

## Performance Considerations

### Current Optimizations
- ✅ NumPy vectorization for PCA
- ✅ Pandas groupby for efficient aggregation
- ✅ statsmodels for optimized ANOVA

### Future Improvements
- 🔜 Cache correlation matrices
- 🔜 Parallel post-hoc tests
- 🔜 Lazy load large XLSX files
- 🔜 Cython for CLD algorithm

## Security & Robustness

### Data Validation
- ✅ Column name sanitization (prevents formula injection)
- ✅ Type coercion with error handling
- ✅ Factor level validation

### Statistical Robustness
- ✅ Fallback ANOVA methods if standard fails
- ✅ Minimum group size checks
- ✅ Proper handling of missing data

## Future Extensions

### Phase 2: UI Components
Move Streamlit-specific code to `ui/`:
```
src/agrecology/ui/
├── __init__.py
├── data_section.py
├── anova_section.py
├── posthoc_section.py
└── visualization_section.py
```

### Phase 3: Advanced Features
- Mixed-effects models
- Bayesian ANOVA
- Effect size calculations
- Power analysis

### Phase 4: Database Integration
- Save analyses to SQLite
- Query historical results
- Compare multiple datasets

## Maintenance Tips

1. **Before changing a function:**
   - Check how many places it's imported
   - Run tests (`pytest`)
   - Update docstrings

2. **Before adding a new constant:**
   - Check if it already exists in `constants.py`
   - Use descriptive names (e.g., `DEFAULT_ANOVA_TYPE` not `TYPE`)

3. **Before refactoring:**
   - Ensure all tests pass
   - Document the change in `MIGRATION_GUIDE.md`
   - Maintain backward compatibility

## References

- **ANOVA**: statsmodels documentation
- **Post-hoc**: scikit-posthocs documentation
- **CLD Algorithm**: Piepho (2004) "An Algorithm for a Letter-Based Representation of All-Pairwise Comparisons"
- **PCA**: NumPy SVD documentation

---

Last Updated: February 2024
