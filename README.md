# Agrecology Analysis Tool

Streamlit + CLI toolkit for agronomy data analysis.

## What It Does

- Load `.csv` or `.xlsx` data (select sheet for Excel).
- Sanitize column names for modeling (spaces/symbols become formula-safe names).
- Define `Rep` as optional blocking factor for RCBD-style ANOVA.
- Select analysis parameter columns automatically (from a start column, numeric-ratio based) or manually.
- Run assumption checks:
  - Shapiro-Wilk normality (overall or by group)
  - Levene homogeneity test
  - QQ plot
- Run ANOVA:
  - Factorial ANOVA (Type I/II/III)
  - Optional block factor (`Rep`)
  - Nested ANOVA (`parent/nested`) with optional block factor
- Run post-hoc:
  - LSD
  - Tukey HSD
  - Bonferroni-adjusted pairwise tests
  - Kruskal-Wallis + Dunn (Bonferroni/Holm/FDR-BH)
- Visualize:
  - Scatter plot
  - Correlation table + heatmap (Pearson/Spearman)
  - PCA biplot (with automatic long-tail handling: optional `log10` before standardization)
  - ANOVA effect plots (interaction lines + mean +/- SD bar charts with CLD letters)
- Export cleaned data as Excel from UI.

## Important Data Assumption (UI)

In `app.py`, the first data row is treated as a units row and removed from analysis.

- Row 1 after header: shown as unit metadata.
- Analysis starts from row 2.

If your file does not contain a units row, remove/adjust this behavior in `app.py`.

## Installation

```bash
pip install -r requirements.txt
```

## Run Streamlit App

```bash
streamlit run app.py
```

Windows helper scripts in this repo:

- `setup_env.ps1`: add local Anaconda path to user `PATH`.
- `activate_env.ps1`: add Anaconda path for current PowerShell session only.
- `run_app.bat`: launch Streamlit app with a fixed local Anaconda Python path.

## CLI Usage

```powershell
python pipeline_cli.py `
  --input your_data.xlsx `
  --sheet-name Sheet1 `
  --response Dry_matter `
  --group Treatment `
  --factors Treatment Soil Biochar Fertilizer `
  --replicate-col Rep `
  --parameter-start-col Dry_matter `
  --nested-parent Soil `
  --nested-child Treatment `
  --outdir outputs
```

CLI outputs:

- `normality.csv`
- `levene.csv`
- `correlation.csv`
- `anova.csv` (if factors or replicate provided)
- `nested_anova.csv` (if nested args provided)
- `lsd.csv`
- `kruskal.csv`
- `dunn.csv`

## Project Files

- `app.py`: Streamlit UI
- `analysis_pipeline.py`: core statistical functions
- `pipeline_cli.py`: command-line runner
- `requirements.txt`: Python dependencies
