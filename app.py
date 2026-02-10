import io

import pandas as pd
import plotly.express as px
import streamlit as st

from analysis_pipeline import (
    anova_analysis,
    correlation_table,
    dunn_posthoc,
    kruskal_wallis,
    levene_homogeneity,
    load_data,
    lsd_posthoc,
    nested_anova,
    normality_checks,
    split_columns,
)

st.set_page_config(page_title="Agrecology 統計分析介面", layout="wide")
st.title("Agrecology 統計分析 Pipeline 與互動介面")
st.caption("支援 ANOVA、巢狀 ANOVA、常態性檢查、LSD / Dunn 後續檢定，以及散佈圖、heatmap、correlation。")

uploaded = st.file_uploader("上傳資料（CSV / XLSX）", type=["csv", "xlsx"])

if not uploaded:
    st.info("請先上傳資料檔。欄位建議包含 Treatment、Rep 與多個數值型態指標。")
    st.stop()

sheet_name = None
if uploaded.name.lower().endswith(".xlsx"):
    xls = pd.ExcelFile(uploaded)
    sheet_name = st.selectbox("選擇工作表", options=xls.sheet_names)
    uploaded.seek(0)

df = load_data(uploaded, sheet_name=sheet_name)
st.subheader("資料預覽")
st.dataframe(df.head(20), use_container_width=True)

numeric_cols, categorical_cols = split_columns(df)

if not numeric_cols:
    st.error("未偵測到數值欄位，無法進行統計分析。")
    st.stop()

if not categorical_cols:
    st.error("未偵測到類別欄位，請確認資料包含 Treatment / Soil 等分組欄位。")
    st.stop()

response = st.selectbox("選擇反應變數（數值）", options=numeric_cols)

with st.expander("資料品質檢查", expanded=False):
    st.write("缺失值統計")
    st.dataframe(df.isna().sum().rename("missing_count"), use_container_width=True)
    st.write("描述統計")
    st.dataframe(df[numeric_cols].describe().T, use_container_width=True)

tab1, tab2, tab3, tab4 = st.tabs(["常態性/前提檢查", "ANOVA", "Post-hoc", "作圖與相關性"])

with tab1:
    st.markdown("### 常態性檢查（Shapiro-Wilk）")
    group_for_norm = st.selectbox("分組欄位（可選）", options=[None] + categorical_cols)
    norm = normality_checks(df, response=response, group=group_for_norm)
    st.dataframe(norm, use_container_width=True)

    st.markdown("### 變異數同質性（Levene）")
    group_for_levene = st.selectbox("Levene 分組欄位", options=categorical_cols)
    lev = levene_homogeneity(df, response=response, group=group_for_levene)
    st.dataframe(lev, use_container_width=True)

with tab2:
    st.markdown("### ANOVA")
    with st.form("anova_form"):
        factors = st.multiselect("選擇因子", options=categorical_cols)
        typ = st.selectbox("ANOVA Type", options=[1, 2, 3], index=1)
        run_anova = st.form_submit_button("執行 ANOVA")

    if run_anova:
        try:
            table = anova_analysis(df, response=response, factors=factors, typ=typ)
            st.dataframe(table, use_container_width=True)
        except Exception as e:
            st.error(f"ANOVA 執行失敗：{e}")

    st.markdown("### 巢狀 ANOVA")
    with st.form("nested_anova_form"):
        c1, c2, c3 = st.columns(3)
        parent = c1.selectbox("上層因子（parent）", options=categorical_cols)
        nested = c2.selectbox("巢狀因子（nested）", options=categorical_cols)
        ntyp = c3.selectbox("ANOVA Type", options=[1, 2, 3], index=1, key="ntyp")
        run_nested = st.form_submit_button("執行巢狀 ANOVA")

    if run_nested:
        if parent == nested:
            st.error("上層因子與巢狀因子不可相同。")
        else:
            try:
                ntable = nested_anova(df, response=response, parent_factor=parent, nested_factor=nested, typ=ntyp)
                st.dataframe(ntable, use_container_width=True)
            except Exception as e:
                st.error(f"巢狀 ANOVA 執行失敗：{e}")

with tab3:
    st.markdown("### Post-hoc 檢定")
    post_group = st.selectbox("分組欄位", options=categorical_cols, key="post_group")
    method = st.radio("方法", options=["LSD", "Dunn"], horizontal=True)
    adjust = st.selectbox("Dunn p-value 校正方式", options=["bonferroni", "holm", "fdr_bh"])

    if st.button("執行 Post-hoc"):
        try:
            if method == "LSD":
                lsd = lsd_posthoc(df, response=response, group=post_group)
                st.dataframe(lsd, use_container_width=True)
            else:
                kw = kruskal_wallis(df, response=response, group=post_group)
                st.write("Kruskal-Wallis 結果")
                st.dataframe(kw, use_container_width=True)
                dunn = dunn_posthoc(df, response=response, group=post_group, p_adjust=adjust)
                st.write("Dunn 檢定 p-value matrix")
                st.dataframe(dunn, use_container_width=True)
        except Exception as e:
            st.error(f"Post-hoc 執行失敗：{e}")

with tab4:
    st.markdown("### 散佈圖")
    c1, c2, c3 = st.columns(3)
    x_col = c1.selectbox("X", options=numeric_cols, key="x_col")
    y_col = c2.selectbox("Y", options=numeric_cols, key="y_col", index=min(1, len(numeric_cols) - 1))
    color_col = c3.selectbox("顏色分組", options=[None] + categorical_cols, key="color_col")
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, hover_data=df.columns)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Correlation / Heatmap")
    corr_method = st.radio("相關係數方法", ["pearson", "spearman"], horizontal=True)
    corr = correlation_table(df, method=corr_method)
    st.dataframe(corr, use_container_width=True)
    hmap = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    st.plotly_chart(hmap, use_container_width=True)

st.markdown("---")
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="cleaned_data", index=False)
st.download_button(
    "下載清理後資料（XLSX）",
    data=buffer.getvalue(),
    file_name="cleaned_data.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
