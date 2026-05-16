import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT_DIR / "weather_classification_data.csv"
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "weather_xgboost.joblib"
DEFAULT_METRICS_PATH = ROOT_DIR / "models" / "metrics.json"

TARGET_COLUMN = "Weather Type"
NUMERIC_FEATURES = [
    "Temperature",
    "Humidity",
    "Wind Speed",
    "Precipitation (%)",
    "Atmospheric Pressure",
    "UV Index",
    "Visibility (km)",
]
CATEGORICAL_FEATURES = ["Cloud Cover", "Season", "Location"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

BASE_XGB_PARAMS = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "random_state": 42,
    "n_jobs": -1,
}

PARAM_DISTRIBUTION = {
    "model__n_estimators": [150, 250, 300, 400],
    "model__max_depth": [3, 4, 5, 6, 7],
    "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
    "model__subsample": [0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.8, 0.9, 1.0],
    "model__min_child_weight": [1, 3, 5],
}


def load_dataset(data_path: Path) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    missing_columns = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in data.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")
    return data[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()


def build_pipeline(num_classes: int, model_params: dict | None = None) -> Pipeline:
    params = BASE_XGB_PARAMS.copy()
    if model_params:
        params.update(model_params)
    params["num_class"] = num_classes

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", XGBClassifier(**params)),
        ]
    )


def get_feature_metadata(data: pd.DataFrame) -> dict:
    numeric = {}
    for column in NUMERIC_FEATURES:
        series = data[column]
        numeric[column] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
        }

    categorical = {
        column: sorted(data[column].dropna().astype(str).unique().tolist())
        for column in CATEGORICAL_FEATURES
    }

    return {
        "numeric": numeric,
        "categorical": categorical,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
    }


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test_encoded, label_encoder: LabelEncoder) -> dict:
    y_pred_encoded = model.predict(x_test)
    target_names = label_encoder.classes_.tolist()

    return {
        "accuracy": float(accuracy_score(y_test_encoded, y_pred_encoded)),
        "f1_macro": float(f1_score(y_test_encoded, y_pred_encoded, average="macro")),
        "precision_macro": float(precision_score(y_test_encoded, y_pred_encoded, average="macro")),
        "recall_macro": float(recall_score(y_test_encoded, y_pred_encoded, average="macro")),
        "classification_report": classification_report(
            y_test_encoded,
            y_pred_encoded,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test_encoded, y_pred_encoded).tolist(),
    }


def train_model(data_path: Path, model_path: Path, metrics_path: Path, tune: bool, n_iter: int) -> dict:
    data = load_dataset(data_path)
    x = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    pipeline = build_pipeline(num_classes=len(label_encoder.classes_))
    best_params = None

    if tune:
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=PARAM_DISTRIBUTION,
            n_iter=n_iter,
            scoring="f1_macro",
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            random_state=42,
            n_jobs=1,
            verbose=1,
        )
        search.fit(x_train, y_train)
        pipeline = search.best_estimator_
        best_params = search.best_params_
    else:
        pipeline.fit(x_train, y_train)

    metrics = evaluate_model(pipeline, x_test, y_test, label_encoder)
    metrics.update(
        {
            "model": "XGBClassifier",
            "tuned": tune,
            "best_params": best_params,
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "classes": label_encoder.classes_.tolist(),
        }
    )

    artifact = {
        "pipeline": pipeline,
        "label_encoder": label_encoder,
        "feature_metadata": get_feature_metadata(data),
        "metrics": metrics,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost untuk klasifikasi tipe cuaca.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--tune", action="store_true", help="Aktifkan RandomizedSearchCV.")
    parser.add_argument("--n-iter", type=int, default=12, help="Jumlah kombinasi tuning saat --tune aktif.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_model(
        data_path=args.data,
        model_path=args.model_out,
        metrics_path=args.metrics_out,
        tune=args.tune,
        n_iter=args.n_iter,
    )

    print("Training selesai.")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-macro: {metrics['f1_macro']:.4f}")
    if metrics["best_params"]:
        print(f"Best params: {metrics['best_params']}")


if __name__ == "__main__":
    main()
