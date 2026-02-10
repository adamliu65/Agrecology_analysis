# Agrecology_analysis

這個專案提供可上傳試算表的統計分析流程，並可指定欄位角色：

- `Rep` 作為**重複/區集（blocking factor）**，用於統計模型而不是分析參數。
- 分析參數可指定「**從某欄開始**」（例如從 `Dry_matter` 開始）或手動選欄。

## 功能

- 資料匯入：支援 `.csv` / `.xlsx`，欄名自動清理
- 前提檢查：Shapiro-Wilk、Levene
- ANOVA：多因子（Type I/II/III）+ 可加入 Rep block
- 巢狀 ANOVA：`parent / nested` + 可加入 Rep block
- Post-hoc：LSD、Kruskal-Wallis + Dunn
- 作圖：scatter、correlation、heatmap

## 安裝

```bash
pip install -r requirements.txt
```

## 啟動介面

```bash
streamlit run app.py
```

## 執行批次 pipeline（CLI）

```bash
python pipeline_cli.py \
  --input your_data.xlsx \
  --sheet-name Sheet1 \
  --response Dry_matter_mg_g_1 \
  --group Treatment \
  --factors Treatment Soil Biochar Fertilizer \
  --nested-parent Soil \
  --nested-child Treatment \
  --outdir outputs
```

CLI 會在 `outputs/` 產生 `normality.csv`、`levene.csv`、`anova.csv`、`nested_anova.csv`、`lsd.csv`、`kruskal.csv`、`dunn.csv`、`correlation.csv`。
