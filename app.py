import io

import pandas as pd
import plotly.express as px
import streamlit as st

from analysis_pipeline import (
    anova_analysis,
    coerce_numeric_columns,
    correlation_table,
    dunn_posthoc,
    kruskal_wallis,
    levene_homogeneity,
    load_data,
    lsd_posthoc,
    nested_anova,
    normality_checks,
    select_parameter_columns,
    split_columns,
)

st.set_page_config(page_title="Agrecology 統計分析介面", layout="wide")
st.title("Agrecology 統計分析 Pipeline 與互動介面")
st.caption("可指定 Rep 為重複欄位（統計 block），並設定分析參數起始欄位（例如從 Dry_matter 開始）。")

uploaded = st.file_uploader("上傳資料（CSV / XLSX）", type=["csv", "xlsx"])
if not uploaded:
    st.info("請先上傳資料檔。")
    st.stop()

sheet_name = None
if uploaded.name.lower().endswith(".xlsx"):
    xls = pd.ExcelFile(uploaded)
    sheet_name = st.selectbox("選擇工作表", options=xls.sheet_names)
    uploaded.seek(0)

df_raw = load_data(uploaded, sheet_name=sheet_name)
all_cols = df_raw.columns.tolist()

st.subheader("欄位角色設定")
c1, c2 = st.columns(2)
replicate_col = c1.selectbox("重複欄位（Rep，可選）", options=[None] + all_cols)
param_mode = c2.radio("參數欄位指定方式", options=["從某欄開始", "手動選擇"], horizontal=True)

if param_mode == "從某欄開始":
    default_start = "Dry_matter" if "Dry_matter" in all_cols else all_cols[min(5, len(all_cols) - 1)]
    parameter_start = st.selectbox("參數起始欄位", options=all_cols, index=all_cols.index(default_start))
    parameter_cols = select_parameter_columns(df_raw, start_col=parameter_start, exclude_cols=[replicate_col] if replicate_col else [])
else:
    parameter_cols = st.multiselect("手動選擇參數欄位", options=all_cols)

if not parameter_cols:
    st.error("未找到可分析的參數欄位，請調整『參數起始欄位』或改用手動選擇。")
    st.stop()

df = coerce_numeric_columns(df_raw, parameter_cols)
numeric_cols, categorical_cols = split_columns(df)

# 因子欄位應排除參數欄位
factor_cols = [c for c in all_cols if c not in parameter_cols]
if replicate_col and replicate_col in parameter_cols:
    factor_cols = [c for c in factor_cols if c != replicate_col]

st.subheader("資料預覽")
st.dataframe(df.head(20), use_container_width=True)
st.write(f"分析參數欄位數：{len(parameter_cols)}")

response = st.selectbox("選擇反應變數（只顯示參數欄）", options=parameter_cols)

tab1, tab2, tab3, tab4 = st.tabs(["常態性/前提檢查", "ANOVA", "Post-hoc", "作圖與相關性"])

with tab1:
    group_for_norm = st.selectbox("常態性分組欄位（可選）", options=[None] + factor_cols)
    st.dataframe(normality_checks(df, response=response, group=group_for_norm), use_container_width=True)

    group_for_levene = st.selectbox("Levene 分組欄位", options=factor_cols)
    st.dataframe(levene_homogeneity(df, response=response, group=group_for_levene), use_container_width=True)

with tab2:
    with st.form("anova_form"):
        factors = st.multiselect("ANOVA 因子", options=[c for c in factor_cols if c != replicate_col])
        typ = st.selectbox("ANOVA Type", options=[1, 2, 3], index=1)
        run_anova = st.form_submit_button("執行 ANOVA")
    if run_anova:
        try:
            st.dataframe(anova_analysis(df, response=response, factors=factors, typ=typ, block_factor=replicate_col), use_container_width=True)
        except Exception as e:
            st.error(f"ANOVA 執行失敗：{e}")

    with st.form("nested_anova_form"):
        c1, c2, c3 = st.columns(3)
        parent = c1.selectbox("上層因子", options=[c for c in factor_cols if c != replicate_col], key="parent")
        nested = c2.selectbox("巢狀因子", options=[c for c in factor_cols if c != replicate_col], key="nested")
        ntyp = c3.selectbox("ANOVA Type", options=[1, 2, 3], index=1)
        run_nested = st.form_submit_button("執行巢狀 ANOVA")
    if run_nested:
        if parent == nested:
            st.error("上層因子與巢狀因子不可相同。")
        else:
            try:
                st.dataframe(
                    nested_anova(
                        df,
                        response=response,
                        parent_factor=parent,
                        nested_factor=nested,
                        typ=ntyp,
                        block_factor=replicate_col,
                    ),
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"巢狀 ANOVA 執行失敗：{e}")

with tab3:
    post_group = st.selectbox("Post-hoc 分組欄位", options=[c for c in factor_cols if c != replicate_col])
    method = st.radio("方法", options=["LSD", "Dunn"], horizontal=True)
    adjust = st.selectbox("Dunn 校正", options=["bonferroni", "holm", "fdr_bh"])

    if st.button("執行 Post-hoc"):
        if method == "LSD":
            st.dataframe(lsd_posthoc(df, response=response, group=post_group), use_container_width=True)
        else:
            st.dataframe(kruskal_wallis(df, response=response, group=post_group), use_container_width=True)
            st.dataframe(dunn_posthoc(df, response=response, group=post_group, p_adjust=adjust), use_container_width=True)

with tab4:
    c1, c2, c3 = st.columns(3)
    x_col = c1.selectbox("X", options=parameter_cols, key="x_col")
    y_col = c2.selectbox("Y", options=parameter_cols, key="y_col", index=min(1, len(parameter_cols) - 1))
    color_col = c3.selectbox("顏色分組", options=[None] + factor_cols)
    st.plotly_chart(px.scatter(df, x=x_col, y=y_col, color=color_col, hover_data=all_cols), use_container_width=True)

    corr_method = st.radio("相關係數方法", ["pearson", "spearman"], horizontal=True)
    corr = correlation_table(df, method=corr_method, columns=parameter_cols)
    st.dataframe(corr, use_container_width=True)
    st.plotly_chart(px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1), use_container_width=True)

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
