# Agrecology Analysis Tool

A comprehensive statistical analysis toolkit for agronomy research with both interactive Streamlit UI and command-line interfaces.

**Key Features:**
- ✅ Statistical tests (ANOVA, Nested ANOVA, Kruskal-Wallis, normality, homogeneity)
- ✅ Post-hoc analysis (LSD, Tukey HSD, Bonferroni, Dunn with multiple correction methods)
- ✅ Rich visualizations (PCA biplot, correlation heatmaps, effect plots with CLD letters)
- ✅ Flexible data input (CSV, XLSX with sheet selection)
- ✅ Automatic data cleaning and column name sanitization
- ✅ Modular, well-documented codebase

## Project Structure

```
agrecology-analysis/
├── src/agrecology/              # Main package (importable as agrecology)
│   ├── __init__.py             # Package exports
│   ├── constants.py            # Magic numbers and configuration constants
│   ├── data_loader.py          # Data I/O and preprocessing
│   ├── statistical_analysis.py # Core stats functions (ANOVA, tests)
│   ├── posthoc_tests.py        # Post-hoc pairwise comparisons
│   ├── visualization.py        # Plotly figures (PCA, QQ plot, layouts)
│   ├── cld_utils.py            # Compact Letter Display generation
│   └── ui/                     # Streamlit UI components (future)
├── config/                      # Configuration files (future)
├── tests/                       # Unit tests (pytest)
├── docs/                        # Documentation and examples
├── app.py                       # Streamlit UI entry point
├── pipeline_cli.py              # CLI entry point
├── pyproject.toml              # Project metadata & dependencies
├── requirements.txt            # Pinned package versions
└── README.md                   # This file
```

## Installation

### Option 1: Development (from source)
```bash
# Clone repository
git clone https://github.com/yourusername/agrecology-analysis.git
cd agrecology-analysis

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install in development mode
pip install -e ".[dev]"
```

### Option 2: Production (from requirements)
```bash
pip install -r requirements.txt
```

## Usage

### Streamlit UI (Interactive)

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

**Workflow:**
1. Upload CSV/XLSX file
2. Select sheet (if XLSX) and configure columns
3. Choose response and parameter variables
4. Run assumption checks (normality, homogeneity)
5. Perform ANOVA with optional blocking factor
6. Execute post-hoc tests (LSD, Tukey, Dunn, etc.)
7. Generate visualizations (PCA, effect plots, correlations)
8. Download cleaned data

**Important:** The UI assumes the first data row contains units and skips it from analysis. Adjust `ASSUMED_UNITS_ROW` in [src/agrecology/constants.py](src/agrecology/constants.py) if needed.

### Command-Line Interface (Batch Processing)

```powershell
python pipeline_cli.py `
  --input data.xlsx `
  --sheet-name Sheet1 `
  --response Dry_matter `
  --group Treatment `
  --factors Treatment Soil Biochar `
  --replicate-col Rep `
  --parameter-start-col Dry_matter `
  --outdir ./outputs
```

**Parameters:**
- `--input` (required): CSV or XLSX file path
- `--response` (required): Numeric response column name
- `--group` (required): Grouping column for post-hoc tests
- `--factors`: Space-separated factor names for ANOVA
- `--replicate-col`: Optional blocking/replication column
- `--parameter-start-col`: Start of numeric parameter columns
- `--nested-parent`, `--nested-child`: For nested ANOVA
- `--sheet-name`: XLSX sheet name (default: first sheet)
- `--outdir`: Output directory for CSV reports (default: outputs/)

**Outputs:**
- `normality.csv` - Shapiro-Wilk test results
- `levene.csv` - Levene homogeneity test
- `correlation.csv` - Pearson/Spearman correlations
- `anova.csv` - ANOVA table
- `nested_anova.csv` - Nested ANOVA results
- `lsd.csv` - LSD post-hoc pairwise comparisons
- `kruskal.csv` - Kruskal-Wallis test
- `dunn.csv` - Dunn post-hoc results

## API Reference

### Data Loading
```python
from agrecology import load_data, split_columns, coerce_numeric_columns

df = load_data(uploaded_file)  # Streamlit UploadedFile
numeric_cols, categorical_cols = split_columns(df)
df_numeric = coerce_numeric_columns(df, numeric_cols)
```

### Statistical Analysis
```python
from agrecology import (
    normality_checks, levene_homogeneity, anova_analysis,
    nested_anova, kruskal_wallis, correlation_table
)

# Check assumptions
normality_tests = normality_checks(df, response='Yield', group='Treatment')
homogeneity_tests = levene_homogeneity(df, response='Yield', group='Treatment')

# Run ANOVA
anova_table = anova_analysis(
    df, response='Yield', factors=['Treatment', 'Fertilizer'], block_factor='Rep'
)

# Non-parametric alternative
kw_result = kruskal_wallis(df, response='Yield', group='Treatment')
```

### Post-hoc Tests
```python
from agrecology import lsd_posthoc, tukey_posthoc, bonferroni_posthoc, dunn_posthoc

# Parametric tests
lsd_result = lsd_posthoc(df, response='Yield', group='Treatment')
tukey_result = tukey_posthoc(df, response='Yield', group='Treatment')

# Non-parametric
dunn_result = dunn_posthoc(df, response='Yield', group='Treatment', p_adjust='bonferroni')
```

### Visualizations
```python
from agrecology import qqplot_figure, pca_biplot_2d, apply_paper_layout

# QQ plot for normality assessment
qq_fig = qqplot_figure(df, response='Yield', group='Treatment')

# PCA with automatic log-transformation of skewed variables
pca_fig, axes_table, vectors_table, error = pca_biplot_2d(
    df, columns=['Yield', 'Height', 'Biomass'], color_col='Treatment'
)

# Apply consistent publication-ready styling
fig = apply_paper_layout(fig, title="My Plot", x_title="X", y_title="Y")
```

## Architecture & Design

### Modular Organization
The codebase is organized into focused modules:
- **data_loader.py**: File I/O, sanitization, column selection
- **statistical_analysis.py**: ANOVA, assumption tests
- **posthoc_tests.py**: Pairwise comparison methods
- **visualization.py**: Plotly figures and styling
- **cld_utils.py**: Piepho's CLD algorithm
- **constants.py**: All magic numbers in one place

This structure ensures:
- ✅ Easy to test individual components
- ✅ Functions can be imported for custom workflows
- ✅ Changes to constants affect all modules
- ✅ Clear separation of concerns

### Key Design Decisions
1. **Constants Centralization**: All default values (alpha=0.05, ANOVA type, etc.) are in constants.py
2. **Type Hints**: Encouraged for clarity (not strictly enforced yet)
3. **Docstrings**: Google-style docstrings for all public functions
4. **Error Handling**: Graceful fallbacks (e.g., robust ANOVA if standard fails)
5. **Publication-Ready**: apply_paper_layout ensures consistent figure styling

## Configuration

All statistical constants are in [src/agrecology/constants.py](src/agrecology/constants.py):

```python
DEFAULT_ALPHA = 0.05  # Significance level
DEFAULT_ANOVA_TYPE = 2  # Sum of squares type (1, 2, or 3)
PCA_SKEW_THRESHOLD = 1.0  # Trigger log-transform for highly skewed data
PCA_RATIO_THRESHOLD = 3.0  # Q95/Q50 ratio threshold
```

Modify these values to customize behavior project-wide.

## Data Format Requirements

**Input File:**
- First row: Column headers (will be sanitized)
- Second row (UI only): Units (will be skipped from analysis)
- Remaining rows: Data

**Column Names:**
- Spaces, hyphens, and special characters are automatically converted to underscores
- Example: "Dry Matter (%)" → "Dry_Matter"

## Common Use Cases

### 1. Simple RCBD ANOVA
```powershell
python pipeline_cli.py `
  --input soil_data.xlsx `
  --response pH `
  --group Treatment `
  --factors Treatment `
  --replicate-col Rep `
  --outdir results/
```

### 2. Factorial ANOVA with Post-hoc
```bash
streamlit run app.py
# Then in UI:
# - Upload data
# - Select factors: Treatment, Fertilizer
# - Run ANOVA
# - Run Tukey HSD post-hoc
```

### 3. Non-parametric Analysis
```powershell
python pipeline_cli.py `
  --input data.csv `
  --response Yield `
  --group Treatment `
  --outdir results/
# Then use kruskal.csv and dunn.csv outputs
```

## Troubleshooting

**Error: "Insufficient valid factors for ANOVA"**
- Ensure factor columns have at least 2 levels
- Check for completely missing values in factor columns

**PCA dimension reduction "needs at least 2 numeric columns"**
- Select at least 2 numeric variables for PCA
- Ensure columns contain numeric data (not text)

**Post-hoc tests show "insufficient groups"**
- Post-hoc tests require at least 2 groups
- Check your grouping variable has multiple levels

## Development

This project uses modern Python development practices:
- **Code style**: Black (configured in pyproject.toml)
- **Import sorting**: isort
- **Type hints**: Optional but encouraged
- **Testing**: pytest with coverage targets

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Run tests
pytest --cov=src/agrecology

# Type checking (optional)
mypy src/agrecology
```

## License

MIT License - see LICENSE file for details

## Citation

If you use this tool in research, please cite:

```bibtex
@software{agrecology2026,
  title = {Agrecology Analysis Tool},
  author = {Adam Liu},
  year = {2026},
  url = {https://github.com/adamliu65/agrecology-analysis}
}
```

## Windows Helper Scripts

The repo includes helper scripts for Windows users:
- `setup_env.ps1`: Add Anaconda to user PATH (permanent)
- `activate_env.ps1`: Activate Anaconda for current session
- `run_app.bat`: Launch Streamlit with fixed Anaconda path

## Support

For issues, questions, or feature requests, please visit the GitHub Issues page.

---

**Last Updated:** February 2026  
**Version:** 1.0.0
