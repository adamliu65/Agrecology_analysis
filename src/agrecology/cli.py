"""
Command-line interface for batch processing of agronomy data analysis.
"""

import argparse
from pathlib import Path

from src.agrecology import (
    anova_analysis,
    coerce_numeric_columns,
    correlation_table,
    dunn_posthoc,
    kruskal_wallis,
    levene_homogeneity,
    load_data_from_path,
    lsd_posthoc,
    nested_anova,
    normality_checks,
    select_parameter_columns,
)


def main() -> None:
    """
    Main CLI entry point for batch analysis.
    
    Processes data file and generates statistical analysis reports.
    """
    parser = argparse.ArgumentParser(
        description="Agrecology analysis pipeline - Statistical analysis for agronomy data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline_cli.py --input data.xlsx --response Yield --group Treatment \\
    --factors Treatment Fertilizer --replicate-col Rep --outdir results/
  
  python pipeline_cli.py --input data.csv --response pH --group Soil \\
    --nested-parent Soil --nested-child Treatment --outdir results/
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--input",
        required=True,
        help="Path to CSV or XLSX file"
    )
    parser.add_argument(
        "--response",
        required=True,
        help="Numeric response variable column name"
    )
    parser.add_argument(
        "--group",
        required=True,
        help="Grouping column name (for post-hoc tests)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--factors",
        nargs="+",
        default=[],
        help="ANOVA factors (space-separated)"
    )
    parser.add_argument(
        "--replicate-col",
        default=None,
        help="Blocking/replication column (optional, e.g., Rep)"
    )
    parser.add_argument(
        "--parameter-start-col",
        default=None,
        help="Start parameter selection from this column"
    )
    parser.add_argument(
        "--nested-parent",
        default=None,
        help="Parent factor for nested ANOVA"
    )
    parser.add_argument(
        "--nested-child",
        default=None,
        help="Child factor for nested ANOVA"
    )
    parser.add_argument(
        "--sheet-name",
        default=None,
        help="Sheet name for XLSX files (default: first sheet)"
    )
    parser.add_argument(
        "--outdir",
        default="outputs",
        help="Output directory for reports (default: outputs/)"
    )
    parser.add_argument(
        "--anova-type",
        type=int,
        choices=[1, 2, 3],
        default=2,
        help="ANOVA sum of squares type (default: 2)"
    )
    
    args = parser.parse_args()

    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    print(f"Loading data from: {args.input}")
    df = load_data_from_path(args.input, sheet_name=args.sheet_name)
    
    parameter_cols = select_parameter_columns(
        df,
        start_col=args.parameter_start_col,
        exclude_cols=[args.replicate_col] if args.replicate_col else [],
    )
    
    if args.response not in parameter_cols:
        raise ValueError(
            f"Response '{args.response}' not found in parameter columns. "
            f"Available: {parameter_cols[:10]}..."
        )

    df = coerce_numeric_columns(df, parameter_cols)
    print(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
    print(f"Parameter columns: {len(parameter_cols)}")

    # Run assumption checks
    print("\nRunning assumption checks...")
    normality_checks(df, response=args.response, group=args.group).to_csv(
        outdir / "normality.csv", index=False
    )
    levene_homogeneity(df, response=args.response, group=args.group).to_csv(
        outdir / "levene.csv", index=False
    )
    correlation_table(df, columns=parameter_cols).to_csv(
        outdir / "correlation.csv"
    )
    print(f"✓ Normality and correlation tests complete")

    # Run ANOVA if factors specified
    if args.factors or args.replicate_col:
        print("\nRunning ANOVA...")
        anova_analysis(
            df,
            response=args.response,
            factors=args.factors,
            typ=args.anova_type,
            block_factor=args.replicate_col
        ).to_csv(outdir / "anova.csv", index=False)
        print(f"✓ ANOVA complete")

    # Run nested ANOVA if specified
    if args.nested_parent and args.nested_child:
        print("\nRunning nested ANOVA...")
        nested_anova(
            df,
            response=args.response,
            parent_factor=args.nested_parent,
            nested_factor=args.nested_child,
            typ=args.anova_type,
            block_factor=args.replicate_col,
        ).to_csv(outdir / "nested_anova.csv", index=False)
        print(f"✓ Nested ANOVA complete")

    # Run post-hoc tests
    print("\nRunning post-hoc tests...")
    lsd_posthoc(df, response=args.response, group=args.group).to_csv(
        outdir / "lsd.csv", index=False
    )
    kruskal_wallis(df, response=args.response, group=args.group).to_csv(
        outdir / "kruskal.csv", index=False
    )
    dunn_posthoc(df, response=args.response, group=args.group).to_csv(
        outdir / "dunn.csv", index=False
    )
    print(f"✓ Post-hoc tests complete")

    print(f"\n✅ Analysis complete! Reports saved to: {outdir.resolve()}")
    print("\nGenerated files:")
    for csv_file in sorted(outdir.glob("*.csv")):
        print(f"  - {csv_file.name}")


if __name__ == "__main__":
    main()
