from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict, List

import pandas as pd
from django.conf import settings

PALETTE = {
    "navy": "#0B1F3A",
    "gold": "#F1B434",
    "teal": "#1AAAE6",
    "crimson": "#C43F3A",
    "slate": "#233348",
}
PRIORITY_COLORS = {
    "High": {"border": PALETTE["crimson"], "background": "rgba(196,63,58,0.28)"},
    "Moderate": {"border": PALETTE["gold"], "background": "rgba(241,180,52,0.25)"},
    "Routine": {"border": PALETTE["teal"], "background": "rgba(26,170,230,0.25)"},
}
CATEGORY_COLORS = [
    {"border": "#F1B434", "background": "rgba(241,180,52,0.18)"},
    {"border": "#1AAAE6", "background": "rgba(26,170,230,0.16)"},
    {"border": "#C43F3A", "background": "rgba(196,63,58,0.2)"},
    {"border": "#9B59B6", "background": "rgba(155,89,182,0.22)"},
    {"border": "#2ECC71", "background": "rgba(46,204,113,0.2)"},
    {"border": "#FF8C42", "background": "rgba(255,140,66,0.2)"},
]


def _read_json(path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _read_parquet(filename: str) -> pd.DataFrame:
    path = settings.PROCESSED_DATA_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@lru_cache(maxsize=1)
def load_metrics() -> Dict:
    return _read_json(settings.INSPECTION_REPORT_PATH)


@lru_cache(maxsize=1)
def load_daily_volume() -> pd.DataFrame:
    df = _read_parquet("daily_call_volume.parquet")
    if df.empty:
        return df
    return df.sort_values("call_date")


@lru_cache(maxsize=1)
def load_call_type_summary() -> pd.DataFrame:
    return _read_parquet("call_type_summary.parquet")


@lru_cache(maxsize=1)
def load_area_hour() -> pd.DataFrame:
    return _read_parquet("call_volume_area_hour.parquet")


@lru_cache(maxsize=1)
def load_monthly_priority() -> pd.DataFrame:
    return _read_parquet("monthly_priority_volume.parquet")


@lru_cache(maxsize=1)
def load_family_month() -> pd.DataFrame:
    return _read_parquet("call_family_month_summary.parquet")


def build_summary_cards() -> List[Dict[str, str]]:
    metrics = load_metrics()
    insights = metrics.get("insights", {})
    records = metrics.get("records")
    date_min = metrics.get("date_min")
    date_max = metrics.get("date_max")
    top_area = next(iter(metrics.get("top_areas", {}).items()), ("N/A", 0))
    outside_share = insights.get("outside_share")
    code6_share = insights.get("code6_share")
    cards = [
        {
            "title": "Total Calls",
            "value": f"{records:,}" if records else "n/a",
            "caption": "Call-for-service records since Jan 2024",
        },
        {
            "title": "Coverage Window",
            "value": f"{_format_date(date_min)} - {_format_date(date_max)}",
            "caption": "Continuous daily feeds in scope",
        },
        {
            "title": "Busiest Area",
            "value": f"{top_area[0]}",
            "caption": f"{top_area[1]:,} calls logged",
        },
        {
            "title": "Outside-Jurisdiction Load",
            "value": _format_percent(outside_share),
            "caption": "Mutual-aid / outside-LA incidents",
        },
        {
            "title": "CODE 6 Share",
            "value": _format_percent(code6_share),
            "caption": "Investigative / follow-up traffic",
        },
    ]
    cards = [card for card in cards if card["value"] != "n/a"]
    return cards


def build_priority_mix() -> Dict[str, float]:
    df = load_call_type_summary()
    if df.empty:
        return {}
    totals = df.groupby("priority_level")["calls"].sum()
    return (totals / totals.sum()).round(3).to_dict()


def top_call_types(limit: int = 8) -> List[Dict[str, str]]:
    df = load_call_type_summary()
    if df.empty:
        return []
    grouped = (
        df.groupby(["Call_Type_Text", "priority_level"])["calls"]
        .sum()
        .reset_index()
        .sort_values("calls", ascending=False)
        .head(limit)
    )
    total_calls = df["calls"].sum()
    rows = []
    for _, row in grouped.iterrows():
        pct = row["calls"] / total_calls if total_calls else 0
        rows.append(
            {
                "call_type": row["Call_Type_Text"],
                "priority": row["priority_level"],
                "calls": f"{int(row['calls']):,}",
                "share": f"{pct:.1%}",
            }
        )
    return rows


def build_daily_chart() -> Dict:
    df = load_daily_volume()
    if df.empty:
        return {}
    labels = [pd.to_datetime(value).strftime("%Y-%m-%d") for value in df["call_date"]]
    dataset = {
        "label": "Calls per day",
        "data": df["calls"].astype(int).tolist(),
        "borderColor": PALETTE["gold"],
        "backgroundColor": "rgba(241,180,52,0.25)",
        "fill": True,
        "tension": 0.3,
    }
    return {"labels": labels, "datasets": [dataset]}


def build_hourly_chart() -> Dict:
    df = load_area_hour()
    if df.empty:
        return {}
    hourly = (
        df.groupby("call_hour")["calls"].sum().reindex(range(24), fill_value=0)
    )
    dataset = {
        "label": "Calls by hour",
        "data": hourly.astype(int).tolist(),
        "backgroundColor": PALETTE["teal"],
        "borderColor": PALETTE["teal"],
    }
    labels = [f"{hour:02d}:00" for hour in hourly.index]
    return {"labels": labels, "datasets": [dataset]}


def build_area_heatmap(top_n: int = 6) -> Dict[str, List]:
    df = load_area_hour()
    if df.empty:
        return {"areas": [], "hours": [], "matrix": []}
    top_areas = (
        df.groupby("Area_Occ")["calls"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )
    subset = df[df["Area_Occ"].isin(top_areas)]
    pivot = subset.pivot_table(
        index="Area_Occ", columns="call_hour", values="calls", fill_value=0
    ).reindex(top_areas)
    pivot.columns = pivot.columns.astype(int)
    pivot = pivot.reindex(columns=range(24), fill_value=0)
    matrix = pivot.values.tolist()
    return {"areas": top_areas, "hours": list(pivot.columns), "matrix": matrix}


def build_story_spotlights() -> List[Dict[str, str]]:
    insights = load_metrics().get("insights", {})
    if not insights:
        return []
    hourly = insights.get("hourly_peak", {})
    weekday = insights.get("weekday_peak", {})
    spotlights = [
        {
            "title": "Evening workload peaks",
            "detail": f"{hourly.get('hour', 0):02d}:00 leads with {hourly.get('calls', 0):,} calls since 2024.",
        },
        {
            "title": "Friday surge",
            "detail": f"{weekday.get('name', 'N/A')} owns {weekday.get('share', 0):.1%} of demand.",
        },
        {
            "title": "Outside calls",
            "detail": f"{insights.get('outside_share', 0):.1%} of dispatches originate outside LAPD divisions.",
        },
    ]
    return spotlights


def build_monthly_priority_chart() -> Dict:
    df = load_monthly_priority()
    if df.empty or "call_month" not in df.columns:
        return {}
    df = df.copy()
    df["call_month"] = pd.to_datetime(df["call_month"])
    pivot = (
        df.pivot_table(index="call_month", columns="priority_level", values="calls", aggfunc="sum")
        .fillna(0)
        .sort_index()
    )
    labels = [value.strftime("%Y-%m") for value in pivot.index]
    ordered_levels = ["High", "Moderate", "Routine"]
    ordered_levels.extend(level for level in pivot.columns if level not in ordered_levels)
    datasets = []
    for priority in ordered_levels:
        if priority not in pivot.columns:
            continue
        palette = PRIORITY_COLORS.get(priority, {"border": PALETTE["slate"], "background": "rgba(35,51,72,0.3)"})
        datasets.append(
            {
                "label": priority,
                "data": pivot[priority].astype(int).tolist(),
                "borderColor": palette["border"],
                "backgroundColor": palette["background"],
                "fill": True,
                "tension": 0.25,
            }
        )
    return {"labels": labels, "datasets": datasets}


def build_family_trend_chart() -> Dict:
    df = load_family_month()
    if df.empty or "call_month" not in df.columns:
        return {}
    df = df.copy()
    df["call_month"] = pd.to_datetime(df["call_month"])
    pivot = (
        df.pivot_table(index="call_month", columns="Call_Type_Family", values="calls", aggfunc="sum")
        .fillna(0)
        .sort_index()
    )
    labels = [value.strftime("%Y-%m") for value in pivot.index]
    datasets = []
    for idx, family in enumerate(pivot.columns):
        colors = CATEGORY_COLORS[idx % len(CATEGORY_COLORS)]
        datasets.append(
            {
                "label": family,
                "data": pivot[family].astype(int).tolist(),
                "borderColor": colors["border"],
                "backgroundColor": colors["background"],
                "fill": True,
                "tension": 0.25,
            }
        )
    return {"labels": labels, "datasets": datasets}


def dataset_profile() -> Dict[str, object]:
    metrics = load_metrics()
    schema = metrics.get("schema", {})
    rows = schema.get("rows") or metrics.get("records") or 0
    columns = schema.get("columns") or len(schema.get("column_types", {})) or 0
    numeric_cols = schema.get("numeric_columns", [])
    categorical_cols = schema.get("categorical_columns", [])
    datetime_cols = schema.get("datetime_columns", [])
    boolean_cols = schema.get("boolean_columns", [])

    sample_columns = schema.get("sample_columns", [])
    raw_rows = schema.get("sample_rows", [])
    sample_rows = [
        [record.get(col, "") for col in sample_columns]
        for record in raw_rows
    ]

    profile = {
        "rows": rows,
        "rows_display": f"{rows:,}" if rows else "n/a",
        "columns": columns,
        "memory_display": f"{schema.get('memory_mb', 0):,.2f} MB" if schema else "n/a",
        "numeric_count": len(numeric_cols),
        "categorical_count": len(categorical_cols),
        "datetime_count": len(datetime_cols),
        "boolean_count": len(boolean_cols),
        "numeric_columns": numeric_cols[:5],
        "categorical_columns": categorical_cols[:5],
        "datetime_columns": datetime_cols[:5],
        "boolean_columns": boolean_cols[:5],
        "sample_columns": sample_columns,
        "sample_rows": sample_rows,
    }
    return profile


def categorical_rollup(limit: int = 4) -> List[Dict[str, object]]:
    records = load_metrics().get("categorical_overview", [])
    formatted: List[Dict[str, object]] = []
    for record in records:
        families = record.get("top_families", [])[:limit]
        formatted.append(
            {
                "year": record.get("year"),
                "months": record.get("months", []),
                "top_families": [
                    {"name": fam.get("name"), "share": _format_percent(fam.get("share"))}
                    for fam in families
                ],
            }
        )
    return formatted


@lru_cache(maxsize=1)
def load_lab_report() -> Dict[str, Any]:
    path = settings.ML_LAB_PATH
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def lab_headlines() -> Dict[str, str]:
    report = load_lab_report()
    metrics = report.get("metrics", {})
    return {
        "accuracy": _format_percent(metrics.get("accuracy")),
        "macro_f1": _format_percent(metrics.get("macro_f1")),
        "weighted_f1": _format_percent(metrics.get("weighted_f1")),
    }


def lab_metadata() -> Dict[str, str]:
    report = load_lab_report()
    model = report.get("model", {})
    generated_at = report.get("generated_at")
    depth = model.get("max_depth")
    estimators = model.get("estimators")
    return {
        "algorithm": model.get("algorithm", "RandomForestClassifier"),
        "depth": depth if depth is not None else "n/a",
        "estimators": estimators if estimators is not None else "n/a",
        "target": report.get("target", "priority_level"),
        "training_rows": f"{report.get('training_rows', 0):,}",
        "test_rows": f"{report.get('test_rows', 0):,}",
        "generated_at": _format_timestamp(generated_at),
        "notes": report.get("notes", ""),
    }


def lab_feature_importances(limit: int = 10) -> List[Dict[str, object]]:
    report = load_lab_report()
    rows = report.get("feature_importances", [])[:limit]
    return [
        {
            "feature": row.get("feature"),
            "importance": round(float(row.get("importance", 0)) * 100, 2),
        }
        for row in rows
    ]


def lab_feature_chart(limit: int = 10) -> Dict[str, Any]:
    rows = lab_feature_importances(limit)
    if not rows:
        return {}
    labels = [row["feature"] for row in rows]
    data = [row["importance"] for row in rows]
    return {
        "labels": labels,
        "datasets": [
            {
                "label": "Feature importance (%)",
                "data": data,
                "backgroundColor": ["rgba(255,255,255,0.15)"] * len(labels),
                "borderColor": [PALETTE["gold"]] * len(labels),
                "borderWidth": 1.5,
            }
        ],
    }


def lab_confusion_rows() -> List[Dict[str, Any]]:
    report = load_lab_report()
    matrix = report.get("confusion_matrix", {})
    labels = matrix.get("labels", [])
    rows = matrix.get("matrix", [])
    formatted = []
    for label, values in zip(labels, rows):
        formatted.append({"label": label, "values": values})
    return formatted


def lab_classification_rows() -> List[Dict[str, Any]]:
    report = load_lab_report()
    data = report.get("classification_report", {})
    row_order = report.get("confusion_matrix", {}).get("labels", [])
    extras = ["macro avg", "weighted avg"]
    rows = []
    for label in row_order + extras:
        stats = data.get(label)
        if not isinstance(stats, dict):
            continue
        rows.append(
            {
                "label": label.title() if label in extras else label,
                "precision": _format_percent(stats.get("precision")),
                "recall": _format_percent(stats.get("recall")),
                "f1": _format_percent(stats.get("f1-score")),
                "support": int(stats.get("support", 0)),
            }
        )
    return rows


def lab_confusion_labels() -> List[str]:
    report = load_lab_report()
    return report.get("confusion_matrix", {}).get("labels", [])


def lab_feature_catalog() -> Dict[str, List[str]]:
    report = load_lab_report()
    catalog = report.get("feature_catalog", {})
    return {
        "numeric": catalog.get("numeric", []),
        "categorical": catalog.get("categorical", []),
    }


def lab_notes() -> str:
    return lab_metadata().get("notes", "")


def _format_date(value: str | None) -> str:
    if not value:
        return "n/a"
    return pd.to_datetime(value).strftime("%d %b %Y")


def _format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}"


def _format_timestamp(value: str | None) -> str:
    if not value:
        return "n/a"
    return pd.to_datetime(value).strftime("%d %b %Y %H:%M UTC")
