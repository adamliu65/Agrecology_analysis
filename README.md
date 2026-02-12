# Agrecology Analysis Tool

Statistical analysis toolkit for agronomy data, with both Streamlit UI and CLI workflows.

## Features

- Data input: `.csv` / `.xlsx` (with sheet selection)
- Data preprocessing:
  - Column-name sanitization for formula-safe modeling
  - Auto/manual parameter column selection
  - Numeric coercion for analysis columns
- Assumption checks:
  - Shapiro-Wilk normality
  - Levene homogeneity
  - QQ plot
- Statistical models:
  - Factorial ANOVA (Type I/II/III)
  - Nested ANOVA
  - Optional block factor (`Rep`) for RCBD-style analysis
  - Kruskal-Wallis
- Post-hoc tests:
  - LSD
  - Tukey HSD
  - Bonferroni pairwise t-tests
  - Dunn (with p-adjust options)
- Visualization:
  - Scatter plot
  - Correlation table + heatmap
  - PCA biplot (with skew-aware transformation)
  - ANOVA interaction/effect plots with CLD letters

## Project Structure

```text
.
|-- app.py                      # Streamlit entrypoint
|-- runtime.txt                 # Streamlit Cloud Python version pin
|-- src/agrecology/             # Package-based implementation
|   |-- __init__.py
|   |-- cli.py
|   |-- constants.py
|   |-- data_loader.py
|   |-- statistical_analysis.py
|   |-- posthoc_tests.py
|   |-- visualization.py
|   `-- cld_utils.py
|-- tests/
|-- requirements.txt            # Pinned production deps
`-- pyproject.toml              # Package metadata + dev tooling config
```

## Installation

### Production

```bash
pip install -r requirements.txt
```

### Development

```bash
pip install -e ".[dev]"
```

## Usage

### Streamlit UI

```bash
streamlit run app.py
```

Then open `http://localhost:8501`.

Note: current `app.py` behavior assumes the first data row after headers is a units row and excludes it from analysis.

### CLI (package entrypoint)

```powershell
python -m src.agrecology.cli `
  --input your_data.xlsx `
  --response Dry_matter `
  --group Treatment `
  --outdir outputs
```

If installed as a package (`pip install -e .`), you can also use:

```bash
agrecology --input your_data.xlsx --response Dry_matter --group Treatment
```

## Streamlit Cloud Deploy

- Deploy entry file: `app.py`
- Python runtime pin: `runtime.txt` set to `python-3.11`

## CLI Outputs

- `normality.csv`
- `levene.csv`
- `correlation.csv`
- `anova.csv` (when factors or replicate are provided)
- `nested_anova.csv` (when nested args are provided)
- `lsd.csv`
- `kruskal.csv`
- `dunn.csv`

## Windows Helper Scripts

- `setup_env.ps1`: add local Anaconda to user PATH
- `activate_env.ps1`: activate Anaconda paths for current PowerShell session
- `run_app.bat`: run Streamlit with fixed local Anaconda Python path

## License

MIT (see `LICENSE`)
