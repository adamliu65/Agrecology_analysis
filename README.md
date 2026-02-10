# Agrecology_analysis

這個專案提供一個可直接上傳試算表的統計分析流程，針對你提供的欄位格式（例如 Treatment、Rep、Soil、Biochar、Fertilizer 與多種化學性質數值欄位）建立：

1. **互動式介面（Streamlit）**
2. **可批次執行的 pipeline CLI**

## 功能

- **資料匯入**：支援 `.csv` 與 `.xlsx`（欄位名稱自動清理）
- **資料品質檢查**：缺失值統計、描述統計
- **前提檢查**：
  - Shapiro-Wilk 常態性檢查（整體或分組）
  - Levene 變異數同質性檢查
- **ANOVA**：多因子 ANOVA（可選 Type I/II/III，含交互作用）
- **巢狀 ANOVA**：`parent / nested` 結構
- **Post-hoc**：
  - Fisher's **LSD**
  - **Dunn**（bonferroni / holm / fdr_bh）
  - Kruskal-Wallis（搭配 Dunn 的前置檢查）
- **作圖與相關性**：
  - 散佈圖（可依類別著色）
  - Correlation table（Pearson / Spearman）
  - Heatmap

## 安裝

```bash
pip install -r requirements.txt
```

## 啟動互動介面

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
