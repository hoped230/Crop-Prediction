from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from scipy.stats import ttest_rel, wilcoxon
from sklearn.base import clone
from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    top_k_accuracy_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler, PolynomialFeatures, PowerTransformer, RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .transformers import DomainFeatureGenerator


class EncodedTargetClassifier(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"

    def __init__(self, estimator):
        self.estimator = estimator

    def get_params(self, deep=True):
        params = {"estimator": self.estimator}
        if deep and hasattr(self.estimator, "get_params"):
            for key, value in self.estimator.get_params(deep=True).items():
                params[f"estimator__{key}"] = value
        return params

    def set_params(self, **params):
        estimator_params = {}
        for key, value in params.items():
            if key == "estimator":
                self.estimator = value
            elif key.startswith("estimator__"):
                estimator_params[key.split("__", 1)[1]] = value
        if estimator_params and hasattr(self.estimator, "set_params"):
            self.estimator.set_params(**estimator_params)
        return self

    def fit(self, X, y):
        self.classes_ = np.array(sorted(pd.Series(y).unique()))
        self.class_to_index_ = {cls: idx for idx, cls in enumerate(self.classes_)}
        encoded = np.array([self.class_to_index_[value] for value in y])
        self.estimator.fit(X, encoded)
        return self

    def predict(self, X):
        encoded_pred = self.estimator.predict(X)
        encoded_pred = np.asarray(encoded_pred, dtype=int)
        return self.classes_[encoded_pred]

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


def resolve_imputer(best_imputation_method: str):
    mapping = {
        "mean": SimpleImputer(strategy="mean"),
        "median": SimpleImputer(strategy="median"),
        "knn": KNNImputer(n_neighbors=5, weights="distance"),
        "iterative_regression": IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=20,
            random_state=42,
        ),
        "em_gaussian": __import__("src.crop_research.preprocessing", fromlist=["EMImputer"]).EMImputer(max_iter=30),
    }
    return mapping[best_imputation_method]


def build_pipeline(imputer, estimator):
    return Pipeline(
        steps=[
            ("domain", DomainFeatureGenerator()),
            ("imputer", clone(imputer)),
            ("distribution", "passthrough"),
            ("scale", "passthrough"),
            ("poly", "passthrough"),
            ("reduce", "passthrough"),
            ("model", estimator),
        ]
    )


def get_search_space(model_name, estimator, imputer):
    scale_choices = [StandardScaler(), MinMaxScaler(), RobustScaler()]
    distribution_choices = ["passthrough", PowerTransformer(method="yeo-johnson")]
    reduce_choices = ["passthrough", PCA(n_components=0.95, random_state=42)]
    poly_choices = ["passthrough", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)]

    pipe = build_pipeline(imputer, estimator)

    if model_name == "LogisticRegression":
        return pipe, GridSearchCV(
            pipe,
            {
                "distribution": distribution_choices,
                "scale": [StandardScaler(), RobustScaler()],
                "poly": poly_choices,
                "reduce": ["passthrough", PCA(n_components=0.95, random_state=42)],
                "model__C": [1.0],
            },
            cv=3,
            scoring="f1_macro",
            n_jobs=1,
            refit=True,
        )
    if model_name == "DecisionTree":
        return pipe, GridSearchCV(
            pipe,
            {
                "distribution": ["passthrough"],
                "scale": ["passthrough"],
                "poly": ["passthrough"],
                "reduce": ["passthrough"],
                "model__criterion": ["gini", "entropy"],
                "model__max_depth": [None, 10],
                "model__min_samples_leaf": [1, 3],
            },
            cv=3,
            scoring="f1_macro",
            n_jobs=1,
            refit=True,
        )
    if model_name == "RandomForest":
        return pipe, RandomizedSearchCV(
            pipe,
            {
                "distribution": ["passthrough"],
                "scale": ["passthrough"],
                "poly": ["passthrough"],
                "reduce": ["passthrough"],
                "model__n_estimators": [200, 300],
                "model__max_depth": [None, 12],
                "model__min_samples_leaf": [1, 2],
                "model__max_features": ["sqrt"],
            },
            n_iter=4,
            cv=3,
            scoring="f1_macro",
            random_state=42,
            n_jobs=1,
            refit=True,
        )
    if model_name == "KNN":
        return pipe, GridSearchCV(
            pipe,
            {
                "distribution": distribution_choices,
                "scale": [StandardScaler(), MinMaxScaler()],
                "poly": ["passthrough"],
                "reduce": ["passthrough", PCA(n_components=0.95, random_state=42)],
                "model__n_neighbors": [5, 9],
                "model__weights": ["uniform", "distance"],
            },
            cv=3,
            scoring="f1_macro",
            n_jobs=1,
            refit=True,
        )
    if model_name == "GaussianNB":
        return pipe, GridSearchCV(
            pipe,
            {
                "distribution": distribution_choices,
                "scale": [StandardScaler(), RobustScaler()],
                "poly": ["passthrough"],
                "reduce": ["passthrough", PCA(n_components=0.95, random_state=42)],
                "model__var_smoothing": [1e-9, 1e-8],
            },
            cv=3,
            scoring="f1_macro",
            n_jobs=1,
            refit=True,
        )
    if model_name == "SVM":
        return pipe, RandomizedSearchCV(
            pipe,
            {
                "distribution": distribution_choices,
                "scale": [StandardScaler(), RobustScaler()],
                "poly": poly_choices,
                "reduce": ["passthrough", PCA(n_components=0.95, random_state=42)],
                "model__C": [1.0, 3.0],
                "model__kernel": ["rbf"],
                "model__gamma": ["scale"],
            },
            n_iter=4,
            cv=3,
            scoring="f1_macro",
            random_state=42,
            n_jobs=1,
            refit=True,
        )
    if model_name == "GradientBoosting":
        return pipe, RandomizedSearchCV(
            pipe,
            {
                "distribution": ["passthrough", PowerTransformer(method="yeo-johnson")],
                "scale": ["passthrough"],
                "poly": ["passthrough"],
                "reduce": ["passthrough"],
                "model__n_estimators": [150, 250],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            },
            n_iter=4,
            cv=3,
            scoring="f1_macro",
            random_state=42,
            n_jobs=1,
            refit=True,
        )
    if model_name == "XGBoost":
        return pipe, RandomizedSearchCV(
            pipe,
            {
                "distribution": ["passthrough"],
                "scale": ["passthrough", StandardScaler()],
                "poly": ["passthrough"],
                "reduce": ["passthrough"],
                "model__estimator__n_estimators": [150, 220],
                "model__estimator__max_depth": [4, 6],
                "model__estimator__learning_rate": [0.05, 0.1],
                "model__estimator__subsample": [0.9, 1.0],
                "model__estimator__colsample_bytree": [0.9, 1.0],
            },
            n_iter=3,
            cv=3,
            scoring="f1_macro",
            random_state=42,
            n_jobs=1,
            refit=True,
        )
    if model_name == "LightGBM":
        return pipe, RandomizedSearchCV(
            pipe,
            {
                "distribution": ["passthrough"],
                "scale": ["passthrough"],
                "poly": ["passthrough"],
                "reduce": ["passthrough"],
                "model__estimator__n_estimators": [150, 220],
                "model__estimator__learning_rate": [0.05, 0.1],
                "model__estimator__num_leaves": [31, 63],
                "model__estimator__max_depth": [-1, 10],
            },
            n_iter=3,
            cv=3,
            scoring="f1_macro",
            random_state=42,
            n_jobs=1,
            refit=True,
        )
    if model_name == "MLP":
        return pipe, RandomizedSearchCV(
            pipe,
            {
                "distribution": distribution_choices,
                "scale": [StandardScaler(), RobustScaler()],
                "poly": ["passthrough"],
                "reduce": ["passthrough", PCA(n_components=0.95, random_state=42)],
                "model__hidden_layer_sizes": [(100,), (128, 64)],
                "model__alpha": [1e-4, 1e-3],
                "model__learning_rate_init": [0.001],
            },
            n_iter=4,
            cv=3,
            scoring="f1_macro",
            random_state=42,
            n_jobs=1,
            refit=True,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def get_estimators():
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42, n_jobs=1),
        "KNN": KNeighborsClassifier(),
        "GaussianNB": GaussianNB(),
        "SVM": SVC(probability=False, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": EncodedTargetClassifier(
            XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=1,
            )
        ),
        "LightGBM": EncodedTargetClassifier(
            lgb.LGBMClassifier(
                objective="multiclass",
                random_state=42,
                n_jobs=1,
                verbosity=-1,
            )
        ),
        "MLP": MLPClassifier(max_iter=400, random_state=42),
    }


def evaluate_classifier(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        metrics["top3_accuracy"] = float(top_k_accuracy_score(y_test, probabilities, k=3, labels=model.classes_))
        metrics["log_loss"] = float(log_loss(y_test, probabilities, labels=model.classes_))
        lb = LabelBinarizer().fit(y_test)
        y_bin = lb.transform(y_test)
        if y_bin.shape[1] == 1:
            y_bin = np.column_stack([1 - y_bin, y_bin])
        metrics["roc_auc_macro_ovr"] = float(roc_auc_score(y_bin, probabilities, multi_class="ovr", average="macro"))
    return metrics, y_pred


def tune_and_compare_models(X, y, best_imputation_method, output_dir: str | Path):
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    dashboard_dir = output_dir / "dashboard"
    best_model_dir = output_dir / "best_model"
    for folder in [tables_dir, figures_dir, dashboard_dir, best_model_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    imputer = resolve_imputer(best_imputation_method)

    search_results = []
    trained_models = {}
    fold_scores = {}

    for model_name, estimator in get_estimators().items():
        print(f"Tuning {model_name}...")
        pipe, search = get_search_space(model_name, estimator, imputer)
        search.fit(X_train, y_train)
        best_estimator = search.best_estimator_
        cv_eval = cross_validate(
            best_estimator,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "f1_macro": "f1_macro",
                "precision_macro": "precision_macro",
                "recall_macro": "recall_macro",
            },
            return_train_score=True,
            n_jobs=1,
        )

        trained_models[model_name] = best_estimator.fit(X_train, y_train)
        fold_scores[model_name] = cv_eval["test_f1_macro"]

        metrics, _ = evaluate_classifier(trained_models[model_name], X_test, y_test)
        train_acc = float(cv_eval["train_accuracy"].mean())
        val_acc = float(cv_eval["test_accuracy"].mean())

        search_results.append(
            {
                "model": model_name,
                "best_params": str(search.best_params_),
                "cv_accuracy_mean": val_acc,
                "cv_accuracy_std": float(cv_eval["test_accuracy"].std()),
                "cv_f1_macro_mean": float(cv_eval["test_f1_macro"].mean()),
                "cv_f1_macro_std": float(cv_eval["test_f1_macro"].std()),
                "cv_precision_macro_mean": float(cv_eval["test_precision_macro"].mean()),
                "cv_recall_macro_mean": float(cv_eval["test_recall_macro"].mean()),
                "train_accuracy_mean": train_acc,
                "bias_variance_gap": train_acc - val_acc,
                "test_accuracy": metrics["accuracy"],
                "test_f1_macro": metrics["f1_macro"],
                "test_precision_macro": metrics["precision_macro"],
                "test_recall_macro": metrics["recall_macro"],
                "test_balanced_accuracy": metrics["balanced_accuracy"],
                "test_top3_accuracy": metrics.get("top3_accuracy"),
                "test_roc_auc_macro_ovr": metrics.get("roc_auc_macro_ovr"),
                "overfitting_flag": (train_acc - val_acc) > 0.05,
            }
        )

    leaderboard = pd.DataFrame(search_results).sort_values(
        ["test_f1_macro", "cv_f1_macro_mean"], ascending=False
    )

    top3_names = [name for name in leaderboard["model"].tolist() if is_classifier(trained_models[name])][:3]
    top3_proba_names = [name for name in top3_names if hasattr(trained_models[name], "predict_proba")]
    ensemble_candidates = {}
    if len(top3_proba_names) >= 2:
        ensemble_candidates["SoftVoting"] = VotingClassifier(
            estimators=[(name, trained_models[name]) for name in top3_proba_names],
            voting="soft",
            n_jobs=1,
        )
    if len(top3_names) >= 2:
        ensemble_candidates["Stacking"] = StackingClassifier(
            estimators=[(name, trained_models[name]) for name in top3_names],
            final_estimator=LogisticRegression(max_iter=2000),
            stack_method="auto",
            passthrough=False,
            n_jobs=1,
        )

    for ensemble_name, ensemble in ensemble_candidates.items():
        print(f"Training {ensemble_name}...")
        ensemble.fit(X_train, y_train)
        cv_eval = cross_validate(
            ensemble,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "f1_macro": "f1_macro",
                "precision_macro": "precision_macro",
                "recall_macro": "recall_macro",
            },
            return_train_score=True,
            n_jobs=1,
        )
        trained_models[ensemble_name] = ensemble
        fold_scores[ensemble_name] = cv_eval["test_f1_macro"]
        metrics, _ = evaluate_classifier(ensemble, X_test, y_test)
        train_acc = float(cv_eval["train_accuracy"].mean())
        val_acc = float(cv_eval["test_accuracy"].mean())
        leaderboard = pd.concat(
            [
                leaderboard,
                pd.DataFrame(
                    [
                        {
                            "model": ensemble_name,
                            "best_params": "top-3 ensemble from tuned base models",
                            "cv_accuracy_mean": val_acc,
                            "cv_accuracy_std": float(cv_eval["test_accuracy"].std()),
                            "cv_f1_macro_mean": float(cv_eval["test_f1_macro"].mean()),
                            "cv_f1_macro_std": float(cv_eval["test_f1_macro"].std()),
                            "cv_precision_macro_mean": float(cv_eval["test_precision_macro"].mean()),
                            "cv_recall_macro_mean": float(cv_eval["test_recall_macro"].mean()),
                            "train_accuracy_mean": train_acc,
                            "bias_variance_gap": train_acc - val_acc,
                            "test_accuracy": metrics["accuracy"],
                            "test_f1_macro": metrics["f1_macro"],
                            "test_precision_macro": metrics["precision_macro"],
                            "test_recall_macro": metrics["recall_macro"],
                            "test_balanced_accuracy": metrics["balanced_accuracy"],
                            "test_top3_accuracy": metrics.get("top3_accuracy"),
                            "test_roc_auc_macro_ovr": metrics.get("roc_auc_macro_ovr"),
                            "overfitting_flag": (train_acc - val_acc) > 0.05,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    leaderboard = leaderboard.sort_values(["test_f1_macro", "cv_f1_macro_mean"], ascending=False).reset_index(drop=True)
    leaderboard.to_csv(tables_dir / "model_leaderboard.csv", index=False)

    best_model_name = leaderboard.iloc[0]["model"]
    best_model = trained_models[best_model_name]
    best_metrics, best_predictions = evaluate_classifier(best_model, X_test, y_test)
    joblib.dump(best_model, best_model_dir / "best_pipeline.joblib")
    (best_model_dir / "best_model_metadata.json").write_text(
        json.dumps(
            {
                "best_model": best_model_name,
                "best_imputation_method": best_imputation_method,
                "metrics": best_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    save_confusion_matrix(best_predictions, y_test, figures_dir / "confusion_matrix.png")
    save_curve_visuals(trained_models, leaderboard, X_test, y_test, figures_dir, dashboard_dir)
    significance_df = compare_model_significance(fold_scores, leaderboard["model"].tolist())
    significance_df.to_csv(tables_dir / "model_significance_tests.csv", index=False)
    save_dashboard(leaderboard, dashboard_dir)

    report_text = classification_report(y_test, best_predictions, zero_division=0)
    (output_dir / "reports" / "best_model_classification_report.txt").write_text(report_text, encoding="utf-8")

    return {
        "leaderboard": leaderboard,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": best_predictions,
        "best_metrics": best_metrics,
        "trained_models": trained_models,
        "significance_tests": significance_df,
    }


def compare_model_significance(fold_scores, model_order):
    best_model = model_order[0]
    rows = []
    best_scores = np.asarray(fold_scores[best_model])
    for other in model_order[1:]:
        if other not in fold_scores:
            continue
        other_scores = np.asarray(fold_scores[other])
        t_stat, t_p = ttest_rel(best_scores, other_scores)
        try:
            w_stat, w_p = wilcoxon(best_scores, other_scores)
        except ValueError:
            w_stat, w_p = np.nan, np.nan
        rows.append(
            {
                "reference_model": best_model,
                "comparison_model": other,
                "reference_mean_f1": float(best_scores.mean()),
                "comparison_mean_f1": float(other_scores.mean()),
                "paired_t_stat": float(t_stat),
                "paired_t_pvalue": float(t_p),
                "wilcoxon_stat": None if np.isnan(w_stat) else float(w_stat),
                "wilcoxon_pvalue": None if np.isnan(w_p) else float(w_p),
            }
        )
    return pd.DataFrame(rows)


def save_confusion_matrix(y_pred, y_test, output_path: str | Path):
    labels = sorted(pd.Series(y_test).unique())
    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(matrix, cmap="YlGn")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_curve_visuals(trained_models, leaderboard, X_test, y_test, figures_dir: Path, dashboard_dir: Path):
    top_models = leaderboard.head(3)["model"].tolist()
    lb = LabelBinarizer().fit(y_test)
    y_bin = lb.transform(y_test)
    if y_bin.shape[1] == 1:
        y_bin = np.column_stack([1 - y_bin, y_bin])

    roc_fig = go.Figure()
    pr_fig = go.Figure()

    for model_name in top_models:
        model = trained_models[model_name]
        if not hasattr(model, "predict_proba"):
            continue
        proba = model.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_bin.ravel(), proba.ravel())
        precision, recall, _ = precision_recall_curve(y_bin.ravel(), proba.ravel())
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=model_name))
        pr_fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=model_name))

    roc_fig.update_layout(title="Micro-Average ROC Curves", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    pr_fig.update_layout(title="Micro-Average Precision-Recall Curves", xaxis_title="Recall", yaxis_title="Precision")
    roc_fig.write_html(dashboard_dir / "roc_curves.html")
    pr_fig.write_html(dashboard_dir / "precision_recall_curves.html")


def save_dashboard(leaderboard: pd.DataFrame, dashboard_dir: Path):
    bar = px.bar(
        leaderboard,
        x="model",
        y=["test_accuracy", "test_f1_macro"],
        barmode="group",
        title="Model Comparison: Accuracy vs Macro F1",
    )
    bar.write_html(dashboard_dir / "model_comparison.html")

    gap = px.bar(
        leaderboard,
        x="model",
        y="bias_variance_gap",
        color="overfitting_flag",
        title="Bias-Variance Gap by Model",
    )
    gap.write_html(dashboard_dir / "bias_variance_gap.html")


def save_model_comparison_plot(leaderboard: pd.DataFrame, output_path: str | Path):
    plot_df = leaderboard.copy().sort_values("test_accuracy", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 7))
    colors = ["#d6e7c7"] * len(plot_df)
    if len(colors) > 0:
        colors[-1] = "#2f6f4f"

    bars = ax.barh(plot_df["model"], plot_df["test_accuracy"], color=colors, edgecolor="#244534")
    ax.set_title("Model Accuracy Comparison", fontsize=16, fontweight="bold")
    ax.set_xlabel("Test Accuracy")
    ax.set_xlim(max(0.94, plot_df["test_accuracy"].min() - 0.01), 1.0)
    ax.grid(axis="x", linestyle="--", alpha=0.25)

    for bar, value in zip(bars, plot_df["test_accuracy"]):
        ax.text(
            value + 0.0006,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            fontsize=9,
            color="#1d3426",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
