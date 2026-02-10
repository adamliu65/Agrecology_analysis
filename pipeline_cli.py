import argparse
from pathlib import Path

from analysis_pipeline import (
    anova_analysis,
    correlation_table,
    dunn_posthoc,
    kruskal_wallis,
    levene_homogeneity,
    load_data_from_path,
    lsd_posthoc,
    nested_anova,
    normality_checks,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Agrecology analysis pipeline CLI")
    parser.add_argument("--input", required=True, help="CSV or XLSX path")
    parser.add_argument("--response", required=True, help="Numeric response column")
    parser.add_argument("--group", required=True, help="Grouping column for assumptions/posthoc")
    parser.add_argument("--factors", nargs="+", default=[], help="ANOVA factors")
    parser.add_argument("--nested-parent", help="Nested ANOVA parent factor")
    parser.add_argument("--nested-child", help="Nested ANOVA nested factor")
    parser.add_argument("--sheet-name", default=None, help="Sheet name for XLSX")
    parser.add_argument("--outdir", default="outputs", help="Output folder")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data_from_path(args.input, sheet_name=args.sheet_name)

    normality_checks(df, response=args.response, group=args.group).to_csv(outdir / "normality.csv", index=False)
    levene_homogeneity(df, response=args.response, group=args.group).to_csv(outdir / "levene.csv", index=False)
    correlation_table(df).to_csv(outdir / "correlation.csv")

    if args.factors:
        anova_analysis(df, response=args.response, factors=args.factors).to_csv(outdir / "anova.csv", index=False)

    if args.nested_parent and args.nested_child:
        nested_anova(
            df,
            response=args.response,
            parent_factor=args.nested_parent,
            nested_factor=args.nested_child,
        ).to_csv(outdir / "nested_anova.csv", index=False)

    lsd_posthoc(df, response=args.response, group=args.group).to_csv(outdir / "lsd.csv", index=False)
    kruskal_wallis(df, response=args.response, group=args.group).to_csv(outdir / "kruskal.csv", index=False)
    dunn_posthoc(df, response=args.response, group=args.group).to_csv(outdir / "dunn.csv", index=False)

    print(f"Analysis reports were generated in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
