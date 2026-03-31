from __future__ import annotations

import argparse
from pathlib import Path

from src.crop_research.data import build_dataset_summary, get_feature_target, load_dataset, save_dataset_profile
from src.crop_research.interpretability import generate_interpretability_outputs
from src.crop_research.modeling import tune_and_compare_models
from src.crop_research.modeling import save_model_comparison_plot
from src.crop_research.preprocessing import (
    benchmark_dimensionality_reduction,
    benchmark_imputation_methods,
    benchmark_scalers,
    correlation_and_covariance_analysis,
    detect_outliers,
)
from src.crop_research.reporting import write_json_summary, write_markdown_report
from src.crop_research.visualization import save_eda_plots, save_pca_and_cluster_plots


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full crop recommendation research pipeline.")
    parser.add_argument("--data", required=True, help="Path to the crop recommendation CSV.")
    parser.add_argument("--output-dir", default="artifacts/research_run", help="Directory for reports and model artifacts.")
    return parser.parse_args()


def ensure_dirs(base_dir: Path):
    dirs = {
        "base": base_dir,
        "reports": base_dir / "reports",
        "figures": base_dir / "figures",
        "tables": base_dir / "tables",
        "dashboard": base_dir / "dashboard",
        "best_model": base_dir / "best_model",
        "interpretability": base_dir / "interpretability",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    dirs = ensure_dirs(output_dir)

    df = load_dataset(args.data)
    X, y = get_feature_target(df)
    dataset_summary = build_dataset_summary(df)
    save_dataset_profile(df, dirs["best_model"] / "data_profile.json")

    correlation_and_covariance_analysis(df, dirs["tables"])
    outlier_summary = detect_outliers(X, dirs["tables"])
    imputation_table, imputation_stats = benchmark_imputation_methods(X, y, dirs["tables"])
    scaler_table = benchmark_scalers(X, y, dirs["tables"])
    reduction_table, _ = benchmark_dimensionality_reduction(X, y, dirs["tables"])

    save_eda_plots(df, dirs["figures"])
    save_pca_and_cluster_plots(df, dirs["figures"])

    modeling_result = tune_and_compare_models(
        X=X,
        y=y,
        best_imputation_method=imputation_stats["best_method"],
        output_dir=output_dir,
    )
    save_model_comparison_plot(
        modeling_result["leaderboard"],
        dirs["figures"] / "model_accuracy_comparison.png",
    )

    generate_interpretability_outputs(
        best_model=modeling_result["best_model"],
        X_train=modeling_result["X_train"],
        X_test=modeling_result["X_test"],
        y_test=modeling_result["y_test"],
        output_dir=dirs["interpretability"],
    )

    write_markdown_report(
        dirs["reports"] / "research_report.md",
        dataset_summary=dataset_summary,
        imputation_stats=imputation_stats,
        best_scaler_row=scaler_table.iloc[0].to_dict(),
        reduction_row=reduction_table.iloc[0].to_dict(),
        modeling_result=modeling_result,
    )
    write_json_summary(
        dirs["reports"] / "run_summary.json",
        {
            "dataset_summary": dataset_summary,
            "outlier_summary": outlier_summary.to_dict(orient="records"),
            "best_imputation_method": imputation_stats["best_method"],
            "best_model": modeling_result["best_model_name"],
            "best_metrics": modeling_result["best_metrics"],
        },
    )

    print("Research pipeline complete.")
    print(f"Outputs saved to: {output_dir}")
    print(f"Best model: {modeling_result['best_model_name']}")
    print(f"Best macro-F1: {modeling_result['best_metrics']['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
