from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from pandas.api import types as ptypes
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from imblearn.pipeline import Pipeline as ImbPipeline  # type: ignore
    from imblearn.over_sampling import SMOTE  # type: ignore
except Exception:  # pragma: no cover - imblearn optional at runtime
    ImbPipeline = None
    SMOTE = None

log = logging.getLogger(__name__)


@dataclass
class LabConfig:
    target: str
    report_path: Path
    synthetic: str = "auto"  # "auto" | "off" | "smote"
    seed: int = 13
    rf_params: Dict[str, Any] | None = None
    max_rows: int | None = None


def _infer_columns(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    if target not in df.columns:
        raise ValueError(f"Target '{target}' is not a column in the provided DataFrame.")

    drop = {target}
    drop.update({col for col in df.columns if col.lower() in {"id", "uuid", "guid"}})
    features = df.drop(columns=list(drop), errors="ignore")

    numeric = [column for column, dtype in features.dtypes.items() if ptypes.is_numeric_dtype(dtype)]
    categorical = [column for column in features.columns if column not in numeric]

    max_cardinality = 200
    filtered_categorical = []
    for column in categorical:
        try:
            card = features[column].nunique(dropna=False)
        except Exception:
            card = max_cardinality + 1
        if card <= max_cardinality:
            filtered_categorical.append(column)
    categorical = filtered_categorical

    if not numeric and not categorical:
        raise ValueError("No feature columns remain after excluding the target/ID columns.")
    return numeric, categorical


def _build_pipeline(
    numeric: List[str],
    categorical: List[str],
    config: LabConfig,
    use_smote: bool,
) -> Pipeline:
    one_hot_kwargs: Dict[str, Any] = {"handle_unknown": "ignore"}
    try:
        OneHotEncoder(sparse_output=False)
        one_hot_kwargs["sparse_output"] = False
    except TypeError:  # pragma: no cover - compat for older sklearn
        one_hot_kwargs["sparse"] = False

    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(**one_hot_kwargs)),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    rf_params = dict(config.rf_params or {})
    rf_params.setdefault("random_state", config.seed)
    if not use_smote:
        rf_params.setdefault("class_weight", "balanced")

    model = RandomForestClassifier(**rf_params)

    if use_smote:
        if ImbPipeline is None or SMOTE is None:
            raise ImportError("imbalanced-learn is required for SMOTE sampling.")
        pipeline: Pipeline = ImbPipeline(
            steps=[
                ("preprocess", preprocess),
                ("sampler", SMOTE(random_state=config.seed)),
                ("model", model),
            ]
        )
    else:
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("model", model),
            ]
        )
    return pipeline


def _feature_names(preprocess: ColumnTransformer, numeric: List[str], categorical: List[str]) -> List[str]:
    return list(preprocess.get_feature_names_out(numeric + categorical))


def _per_class_metrics(y_true, y_pred, labels: List[str]) -> List[Dict[str, Any]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return [
        {
            "label": str(labels[idx]),
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
        for idx in range(len(labels))
    ]


def _macro_weighted(y_true, y_pred) -> Dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
    }
    macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    metrics.update(
        {
            "precision_macro": float(macro[0]),
            "recall_macro": float(macro[1]),
            "f1_macro": float(macro[2]),
            "precision_weighted": float(weighted[0]),
            "recall_weighted": float(weighted[1]),
            "f1_weighted": float(weighted[2]),
        }
    )
    return {key: float(value) for key, value in metrics.items()}


def generate_lab_report(df: pd.DataFrame, target: str, config: LabConfig) -> Dict[str, Any]:
    numeric, categorical = _infer_columns(df, target)

    working = df.dropna(subset=[target]).copy()
    if config.max_rows and len(working) > config.max_rows:
        working = working.sample(config.max_rows, random_state=config.seed)
    X = working[numeric + categorical]
    y = working[target].astype("category").astype(str)

    labels = sorted(y.unique().tolist())
    class_counts = y.value_counts().to_dict()
    minority_ratio = min(class_counts.values()) / y.shape[0]

    synthetic_policy = (config.synthetic or "auto").lower()
    use_smote = synthetic_policy == "smote" or (
        synthetic_policy == "auto" and minority_ratio < 0.20
    )

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=config.seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.20,
        random_state=config.seed,
        stratify=y_train_val,
    )

    pipeline = _build_pipeline(numeric, categorical, config, use_smote=use_smote)
    pipeline.fit(X_train, y_train)

    y_val_pred = pipeline.predict(X_val)
    y_test_pred = pipeline.predict(X_test)

    metrics_val = _macro_weighted(y_val, y_val_pred)
    metrics_val["by_class"] = _per_class_metrics(y_val, y_val_pred, labels)

    metrics_test = _macro_weighted(y_test, y_test_pred)
    metrics_test["by_class"] = _per_class_metrics(y_test, y_test_pred, labels)

    cm_raw = confusion_matrix(y_test, y_test_pred, labels=labels)
    with np.errstate(all="ignore"):
        cm_norm = cm_raw / cm_raw.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    model = pipeline.named_steps["model"]
    preprocess = pipeline.named_steps["preprocess"]
    feature_names = _feature_names(preprocess, numeric, categorical)
    importances = getattr(model, "feature_importances_", None)
    top_features: List[Dict[str, Any]] = []
    if importances is not None:
        ranked = sorted(zip(feature_names, importances), key=lambda pair: pair[1], reverse=True)
        top_features = [
            {"feature": name, "importance": float(score)} for name, score in ranked[:50]
        ]

    total_rows = int(working.shape[0])
    report: Dict[str, Any] = {
        "meta": {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "sklearn_version": __import__("sklearn").__version__,
            "n_samples": total_rows,
            "n_features": len(numeric) + len(categorical),
            "target": target,
            "class_distribution": {str(label): int(count) for label, count in class_counts.items()},
            "minority_ratio": float(minority_ratio),
            "used_synthetic": bool(use_smote),
            "sampler": "SMOTE" if use_smote else None,
            "notes": [],
        },
        "splits": {
            "train": {"share": round(len(X_train) / total_rows, 3), "rows": int(len(X_train))},
            "val": {"share": round(len(X_val) / total_rows, 3), "rows": int(len(X_val))},
            "test": {"share": round(len(X_test) / total_rows, 3), "rows": int(len(X_test))},
        },
        "metrics": {
            "validation": metrics_val,
            "test": metrics_test,
        },
        "confusion_matrix": {
            "labels": [str(label) for label in labels],
            "raw": cm_raw.tolist(),
            "normalized": cm_norm.tolist(),
        },
        "feature_importances": top_features,
    }

    report_path = Path(config.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    log.info("ML Lab report written to %s", report_path)
    return report
