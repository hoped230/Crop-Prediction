import argparse
import json
from pathlib import Path

import joblib
from src.crop_prediction.data import load_dataset, split_features_target
from src.crop_prediction.models import (
    build_training_summary,
    compare_models,
    evaluate_pipeline,
    write_classification_report,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and compare crop recommendation models."
    )
    parser.add_argument("--data", required=True, help="Path to input CSV file.")
    parser.add_argument(
        "--target",
        default="crop",
        help="Target column name. Default: crop",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Directory to store trained models and reports.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split fraction. Default: 0.2",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Cross-validation folds. Default: 5",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data)
    X, y = split_features_target(df, args.target)

    leaderboard_df, best_result, feature_summary = compare_models(
        X=X,
        y=y,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
    )

    leaderboard_path = artifacts_dir / "leaderboard.csv"
    leaderboard_df.to_csv(leaderboard_path, index=False)

    test_metrics, y_test, y_pred = evaluate_pipeline(best_result)

    model_path = artifacts_dir / "best_pipeline.joblib"
    joblib.dump(best_result["pipeline"], model_path)

    metrics_path = artifacts_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    report_path = artifacts_dir / "classification_report.txt"
    write_classification_report(report_path, y_test, y_pred)

    summary = build_training_summary(
        args=args,
        dataset=df,
        feature_summary=feature_summary,
        best_result=best_result,
        leaderboard=leaderboard_df,
        model_path=model_path,
    )
    summary_path = artifacts_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Display results and paths of saved artifacts to the user

    print("\nModel comparison complete.")
    print(f"Leaderboard saved to: {leaderboard_path}")
    print(f"Best model: {best_result['model_name']}")
    print(f"Saved pipeline to: {model_path}")
    print("\nHold-out test metrics:")
    for metric_name, value in test_metrics.items():
        print(f"  {metric_name}: {value:.4f}")


# Entry point of the script
# Executes main() when the script is run directly

if __name__ == "__main__":
    main()
