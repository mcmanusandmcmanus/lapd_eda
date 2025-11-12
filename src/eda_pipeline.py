from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import OrdinalEncoder

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "LAPD_Calls_for_Service_2024_to_Present_20251110.csv"
INTERIM_PATH = BASE_DIR / "data" / "interim" / "lapd_calls_clean.parquet"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "lapd_calls_features.parquet"
AGG_AREA_HOUR_PATH = BASE_DIR / "data" / "processed" / "call_volume_area_hour.parquet"
DAILY_VOLUME_PATH = BASE_DIR / "data" / "processed" / "daily_call_volume.parquet"
CALL_TYPE_SUMMARY_PATH = BASE_DIR / "data" / "processed" / "call_type_summary.parquet"
MONTHLY_PRIORITY_PATH = BASE_DIR / "data" / "processed" / "monthly_priority_volume.parquet"
CALL_FAMILY_MONTH_PATH = BASE_DIR / "data" / "processed" / "call_family_month_summary.parquet"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
SUMMARY_PATH = BASE_DIR / "reports" / "eda_summary.md"
PROFILE_JSON = BASE_DIR / "reports" / "inspection_metrics.json"
ML_LAB_PATH = BASE_DIR / "reports" / "ml_lab.json"
SAMPLE_COLUMNS = [
    "Incident_Number",
    "Call_Type_Text",
    "priority_level",
    "Area_Occ",
    "Call_Type_Family",
    "call_month",
]


def configure_matplotlib() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({"axes.spines.right": False, "axes.spines.top": False})


def load_raw_data(limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH, nrows=limit)
    return df


def normalize_strings(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().replace({"nan": np.nan})


def parse_dispatch_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Dispatch_Date"] = pd.to_datetime(
        df["Dispatch_Date"],
        format="%Y %b %d %I:%M:%S %p",
        errors="coerce",
    )
    df["Dispatch_Time"] = pd.to_timedelta(df["Dispatch_Time"], errors="coerce")
    df["dispatch_datetime"] = df["Dispatch_Date"] + df["Dispatch_Time"]
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Incident_Number"] = normalize_strings(df["Incident_Number"])
    df["Area_Occ"] = normalize_strings(df["Area_Occ"]).fillna("Unknown")
    df["Rpt_Dist"] = normalize_strings(df["Rpt_Dist"])
    df["Call_Type_Code"] = (
        normalize_strings(df["Call_Type_Code"]).str.upper().fillna("UNKNOWN")
    )
    df["Call_Type_Text"] = (
        normalize_strings(df["Call_Type_Text"])
        .str.title()
        .replace({"": np.nan})
        .fillna("Unspecified")
    )
    df = parse_dispatch_columns(df)
    df = df.drop_duplicates(subset=["Incident_Number"], keep="first")
    return df


def augment_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = df["dispatch_datetime"]
    df["call_date"] = dt.dt.date
    df["call_year"] = dt.dt.year
    df["call_month"] = dt.dt.to_period("M").astype(str)
    df["call_hour"] = dt.dt.hour
    df["call_weekday"] = dt.dt.day_name()
    df["call_week"] = dt.dt.isocalendar().week
    df["day_of_year"] = dt.dt.dayofyear
    df["is_weekend"] = df["call_weekday"].isin(["Saturday", "Sunday"])
    df["daypart"] = df["call_hour"].apply(bucket_hour_to_daypart)
    weekday_code = df["call_weekday"].fillna("UNK").str[:3]
    hour_code = df["call_hour"].fillna(-1).astype(int).astype(str)
    df["week_hour"] = weekday_code + "-" + hour_code
    df["is_peak_hour"] = df["call_hour"].between(15, 22, inclusive="left")
    return df


def bucket_hour_to_daypart(hour: float | int | None) -> str:
    if pd.isna(hour):
        return "Unknown"
    hour = int(hour)
    if 0 <= hour < 6:
        return "Late Night"
    if 6 <= hour < 12:
        return "Morning"
    if 12 <= hour < 18:
        return "Afternoon"
    return "Evening"


def infer_priority(text: str, code: str) -> str:
    text = (text or "").upper()
    code = (code or "").upper()
    high_markers = ["EMERGENCY", "ASSAULT", "SHOTS", "PANIC", "CODE 3", "HELP"]
    medium_markers = ["CODE 2", "FAMILY", "DISTURBANCE", "BURGLARY", "THEFT"]

    if any(marker in text for marker in high_markers) or code.endswith("F"):
        return "High"
    if any(marker in text for marker in medium_markers):
        return "Moderate"
    if code.startswith(("6", "7", "8")):
        return "Moderate"
    return "Routine"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = augment_temporal_features(df)
    df["is_outside_area"] = df["Area_Occ"].eq("Outside")
    df["Call_Type_Family"] = (
        df["Call_Type_Text"]
        .str.replace(r"\d+", "", regex=True)
        .str.replace(r"\s{2,}", " ", regex=True)
        .str.strip()
    )
    df["priority_level"] = [
        infer_priority(text, code)
        for text, code in zip(df["Call_Type_Text"], df["Call_Type_Code"], strict=False)
    ]
    df["dispatch_is_missing"] = df["dispatch_datetime"].isna()
    df["operational_day"] = np.where(
        df["dispatch_datetime"].dt.hour < 6,
        (df["dispatch_datetime"] - pd.Timedelta(days=1)).dt.date,
        df["call_date"],
    )
    df["calls_since_midnight"] = (
        df.sort_values("dispatch_datetime")
        .groupby("call_date")
        .cumcount()
    )
    return df


def compute_quality_metrics(df: pd.DataFrame) -> Dict[str, object]:
    missing = (
        df.isna().mean().sort_values(ascending=False).round(4).to_dict()
    )
    area_counts = (
        df["Area_Occ"]
        .value_counts()
        .head(5)
        .to_dict()
    )
    call_type_counts = (
        df["Call_Type_Text"].value_counts().head(5).to_dict()
    )
    dispatch_valid_share = float((~df["dispatch_datetime"].isna()).mean())
    date_min = str(df["dispatch_datetime"].min())
    date_max = str(df["dispatch_datetime"].max())
    metrics = {
        "records": int(len(df)),
        "unique_incidents": int(df["Incident_Number"].nunique()),
        "date_min": date_min,
        "date_max": date_max,
        "dispatch_valid_share": dispatch_valid_share,
        "missing_fraction": missing,
        "top_areas": area_counts,
        "top_call_types": call_type_counts,
    }
    return metrics


def plot_daily_volume(df: pd.DataFrame) -> Path:
    series = (
        df.dropna(subset=["dispatch_datetime"])
        .groupby(df["dispatch_datetime"].dt.date)
        .size()
    )
    fig, ax = plt.subplots(figsize=(14, 6))
    series.plot(ax=ax, color="#0B5ED7")
    ax.set_title("Daily LAPD Call Volume (2024–Present)")
    ax.set_xlabel("Dispatch Date")
    ax.set_ylabel("Number of Calls")
    ax.grid(True, alpha=0.3)
    path = FIGURES_DIR / "daily_call_volume.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_hourly_distribution(df: pd.DataFrame) -> Path:
    hour_counts = (
        df[df["call_hour"].notna()]
        .groupby("call_hour")
        .size()
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(x=hour_counts.index, y=hour_counts.values, color="#F1B434", ax=ax)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Calls")
    ax.set_title("Calls by Hour of Day")
    path = FIGURES_DIR / "calls_by_hour.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_call_type_distribution(df: pd.DataFrame) -> Path:
    top_call_types = df["Call_Type_Text"].value_counts().head(12)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        y=top_call_types.index,
        x=top_call_types.values,
        color="#1f77b4",
        ax=ax,
    )
    ax.set_title("Top Call Types")
    ax.set_xlabel("Calls")
    ax.set_ylabel("")
    path = FIGURES_DIR / "top_call_types.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_area_distribution(df: pd.DataFrame) -> Path:
    area_counts = df["Area_Occ"].value_counts().head(15).sort_values()
    fig, ax = plt.subplots(figsize=(10, 8))
    area_counts.plot.barh(color="#1F78B4", ax=ax)
    ax.set_title("Call Volume by Reporting Area (Top 15)")
    ax.set_xlabel("Calls")
    path = FIGURES_DIR / "top_areas.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_weekday_hour_heatmap(df: pd.DataFrame) -> Path:
    subset = df[df["call_hour"].notna()]
    pivot = (
        subset.groupby(["call_weekday", "call_hour"])
        .size()
        .reset_index(name="calls")
        .pivot(index="call_weekday", columns="call_hour", values="calls")
        .reindex(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        pivot,
        cmap="mako",
        ax=ax,
        cbar_kws={"label": "Calls"},
    )
    ax.set_title("Call Density by Weekday and Hour")
    ax.set_xlabel("Hour")
    ax.set_ylabel("")
    path = FIGURES_DIR / "weekday_hour_heatmap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_eda_outputs(df: pd.DataFrame) -> Dict[str, str]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    outputs = {
        "daily_call_volume": str(plot_daily_volume(df)),
        "hourly_distribution": str(plot_hourly_distribution(df)),
        "call_type_distribution": str(plot_call_type_distribution(df)),
        "area_distribution": str(plot_area_distribution(df)),
        "weekday_hour_heatmap": str(plot_weekday_hour_heatmap(df)),
    }
    return outputs


def aggregate_area_hour(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.dropna(subset=["call_hour"])
        .groupby(["Area_Occ", "call_hour"])
        .size()
        .reset_index(name="calls")
    )
    total = agg.groupby("Area_Occ")["calls"].transform("sum")
    agg["area_hour_share"] = agg["calls"] / total
    return agg


def aggregate_daily_counts(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.dropna(subset=["dispatch_datetime"])
        .groupby("call_date")
        .size()
        .reset_index(name="calls")
    )
    daily["call_date"] = pd.to_datetime(daily["call_date"])
    return daily


def aggregate_call_types(df: pd.DataFrame) -> pd.DataFrame:
    call_types = (
        df.groupby(["Call_Type_Text", "priority_level"])
        .size()
        .reset_index(name="calls")
        .sort_values("calls", ascending=False)
    )
    return call_types


def aggregate_monthly_priority(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.dropna(subset=["call_month", "priority_level"])
        .groupby(["call_month", "priority_level"])
        .size()
        .reset_index(name="calls")
    )
    if monthly.empty:
        return monthly
    monthly["sort_key"] = pd.to_datetime(monthly["call_month"])
    monthly = monthly.sort_values(["sort_key", "priority_level"]).drop(columns="sort_key")
    return monthly


def aggregate_family_month(df: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    top_families = (
        df["Call_Type_Family"]
        .value_counts()
        .head(top_n)
        .index
    )
    subset = df[df["Call_Type_Family"].isin(top_families)]
    monthly = (
        subset.groupby(["call_month", "Call_Type_Family"])
        .size()
        .reset_index(name="calls")
    )
    if monthly.empty:
        return monthly
    monthly["sort_key"] = pd.to_datetime(monthly["call_month"])
    monthly = monthly.sort_values(["sort_key", "Call_Type_Family"]).drop(columns="sort_key")
    return monthly


def summarize_schema(df: pd.DataFrame, sample_columns: List[str] | None = None) -> Dict[str, object]:
    schema_profile: Dict[str, object] = {}
    sample_columns = sample_columns or []
    schema_profile["rows"] = int(len(df))
    schema_profile["columns"] = int(df.shape[1])
    schema_profile["memory_mb"] = round(float(df.memory_usage(deep=True).sum() / 1_000_000), 2)
    schema_profile["column_types"] = {col: str(dtype) for col, dtype in df.dtypes.items()}

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns, UTC]", "datetime64[ns]"]).columns.tolist()
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()

    schema_profile["numeric_columns"] = numeric_cols
    schema_profile["categorical_columns"] = categorical_cols
    schema_profile["datetime_columns"] = datetime_cols
    schema_profile["boolean_columns"] = bool_cols

    preview_cols = [col for col in (sample_columns or []) if col in df.columns]
    if not preview_cols:
        preview_cols = df.columns[:6].tolist()
    sample_df = (
        df[preview_cols]
        .head(5)
        .fillna("")
        .astype(str)
    )
    schema_profile["sample_columns"] = preview_cols
    schema_profile["sample_rows"] = sample_df.to_dict(orient="records")
    return schema_profile


def build_categorical_overview(df: pd.DataFrame, top_n: int = 4) -> List[Dict[str, object]]:
    overview: List[Dict[str, object]] = []
    if "call_year" not in df.columns or "Call_Type_Family" not in df.columns:
        return overview
    grouped = df.groupby("call_year")
    for year, year_df in grouped:
        if pd.isna(year):
            continue
        counts = (
            year_df["Call_Type_Family"]
            .value_counts(normalize=True)
            .head(top_n)
        )
        year_record = {
            "year": int(year),
            "top_families": [
                {"name": name, "share": round(float(share), 4)}
                for name, share in counts.items()
            ],
            "months": sorted(set(year_df["call_month"].astype(str))),
        }
        overview.append(year_record)
    return sorted(overview, key=lambda item: item["year"])


def run_ml_lab(df: pd.DataFrame, sample_size: int = 200_000) -> None:
    target = "priority_level"
    feature_columns = [
        "call_hour",
        "day_of_year",
        "calls_since_midnight",
        "is_weekend",
        "is_peak_hour",
        "is_outside_area",
        "Call_Type_Family",
        "Area_Occ",
        "Call_Type_Text",
        "daypart",
    ]
    required_columns = feature_columns + [target]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        ML_LAB_PATH.write_text(
            json.dumps({"error": f"Missing columns: {', '.join(missing_cols)}"}, indent=2)
        )
        return

    model_df = df[required_columns].dropna()
    model_df = model_df[model_df[target].isin(["High", "Moderate", "Routine"])]
    if model_df.empty:
        ML_LAB_PATH.write_text(json.dumps({"error": "No rows available for ML lab"}, indent=2))
        return

    if len(model_df) > sample_size:
        model_df = model_df.sample(sample_size, random_state=42)
    model_df = model_df.copy()

    boolean_features = ["is_weekend", "is_peak_hour", "is_outside_area"]
    bool_converted = {
        col: model_df[col].astype(int, copy=False)
        for col in boolean_features
    }
    model_df = model_df.assign(**bool_converted)

    numeric_features = ["call_hour", "day_of_year", "calls_since_midnight"] + boolean_features
    categorical_features = [
        "Call_Type_Family",
        "Area_Occ",
        "Call_Type_Text",
        "daypart",
    ]

    X = model_df[numeric_features + categorical_features]
    y = model_df[target]
    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical_features,
            ),
        ],
        remainder="drop",
    )
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=14,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model = SKPipeline(steps=[("preprocess", preprocess), ("clf", clf)])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = float((y_pred == y_test).mean())
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))
    labels = model.named_steps["clf"].classes_.tolist()
    cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()

    report_raw = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report: Dict[str, object] = {}
    for label, stats in report_raw.items():
        if isinstance(stats, dict):
            report[label] = {
                metric: (round(float(value), 4) if isinstance(value, (int, float)) else value)
                for metric, value in stats.items()
            }
        else:
            report[label] = round(float(stats), 4)
    feature_names = numeric_features + categorical_features
    importances = model.named_steps["clf"].feature_importances_.tolist()
    feature_rows = sorted(
        [
            {"feature": name, "importance": round(float(score), 4)}
            for name, score in zip(feature_names, importances, strict=False)
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )[:15]

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "target": target,
        "model": {
            "algorithm": "RandomForestClassifier",
            "max_depth": 14,
            "estimators": 200,
        },
        "metrics": {
            "accuracy": round(accuracy, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
        },
        "classification_report": report,
        "confusion_matrix": {"labels": labels, "matrix": cm},
        "feature_importances": feature_rows,
        "feature_catalog": {
            "numeric": numeric_features,
            "categorical": categorical_features,
        },
        "notes": "Model trained on sampled LAPD calls to classify priority level. Static dataset snapshot.",
    }
    ML_LAB_PATH.write_text(json.dumps(payload, indent=2))


def build_summary_markdown(metrics: Dict[str, object], insights: Dict[str, object]) -> str:
    first_dispatch = pd.to_datetime(metrics["date_min"])
    last_dispatch = pd.to_datetime(metrics["date_max"])
    horizon_days = (last_dispatch - first_dispatch).days
    daily_counts = insights["daily_counts"]
    hourly_peak = insights["hourly_peak"]
    weekday_peak = insights["weekday_peak"]
    outside_share = insights["outside_share"]
    code6_share = insights["code6_share"]
    md_lines = [
        "# LAPD Calls for Service — Inspection & EDA",
        "",
        "## Dataset Snapshot",
        f"- **Records analysed:** {metrics['records']:,} ({metrics['unique_incidents']:,} unique incidents)",
        f"- **Temporal coverage:** {first_dispatch:%d %b %Y} to {last_dispatch:%d %b %Y} (~{horizon_days} days)",
        f"- **Dispatch timestamps present:** {metrics['dispatch_valid_share']:.1%}",
        f"- **Most active areas:** {format_toplist(metrics['top_areas'])}",
        f"- **Top call types:** {format_toplist(metrics['top_call_types'])}",
        "",
        "## Data Quality Watchlist",
        describe_missing(metrics["missing_fraction"]),
        "",
        "## Operational Highlights",
        f"- Daily call demand ranges between {daily_counts['min']:,} and {daily_counts['max']:,} with a median of {daily_counts['median']:,} calls per day.",
        f"- Peak dispatch hour: {hourly_peak['hour']:02d}:00 with {hourly_peak['calls']:,} calls; quietest hour is {hourly_peak['min_hour']:02d}:00 ({hourly_peak['min_calls']:,} calls).",
        f"- {weekday_peak['name']} carries {weekday_peak['share']:.1%} of all calls, outpacing the slowest day ({weekday_peak['min_name']} at {weekday_peak['min_share']:.1%}).",
        f"- \"Outside\" calls account for {outside_share:.1%} of all dispatches, indicating notable mutual-aid or cross-jurisdiction events.",
        f"- CODE 6 related activity represents {code6_share:.1%} of calls, underscoring the load of investigative/stakeout work.",
        "",
        "## Files Generated",
        f"- Clean call-level parquet: `{INTERIM_PATH}`",
        f"- Feature-enhanced parquet: `{PROCESSED_PATH}`",
        f"- Area-hour aggregate: `{AGG_AREA_HOUR_PATH}`",
        f"- Daily call volume: `{DAILY_VOLUME_PATH}`",
        f"- Call type summary: `{CALL_TYPE_SUMMARY_PATH}`",
        f"- Visuals directory: `{FIGURES_DIR}`",
    ]
    return "\n".join(md_lines)


def format_toplist(counter: Dict[str, int]) -> str:
    items = [f"{k} ({v:,})" for k, v in counter.items()]
    return ", ".join(items)


def describe_missing(missing_fraction: Dict[str, float]) -> str:
    ordered = sorted(missing_fraction.items(), key=lambda kv: kv[1], reverse=True)
    bullets = [
        f"- `{col}` missing {share:.1%}"
        for col, share in ordered
        if share > 0
    ]
    if not bullets:
        return "- No missing data detected."
    return "\n".join(bullets)


def compute_insights(df: pd.DataFrame) -> Dict[str, object]:
    daily_counts = (
        df.dropna(subset=["dispatch_datetime"])
        .groupby(df["dispatch_datetime"].dt.date)
        .size()
        .describe()[["min", "max", "50%"]]
    )
    hourly_counts = df.groupby("call_hour").size()
    weekday_counts = df.groupby("call_weekday").size()
    total_calls = len(df)
    outside_share = df["is_outside_area"].mean()
    code6_share = df["Call_Type_Text"].str.contains("Code 6", case=False, na=False).mean()
    insights = {
        "daily_counts": {
            "min": int(daily_counts["min"]),
            "max": int(daily_counts["max"]),
            "median": int(daily_counts["50%"]),
        },
        "hourly_peak": {
            "hour": int(hourly_counts.idxmax()),
            "calls": int(hourly_counts.max()),
            "min_hour": int(hourly_counts.idxmin()),
            "min_calls": int(hourly_counts.min()),
        },
        "weekday_peak": {
            "name": weekday_counts.idxmax(),
            "share": float(weekday_counts.max() / total_calls),
            "min_name": weekday_counts.idxmin(),
            "min_share": float(weekday_counts.min() / total_calls),
        },
        "outside_share": float(outside_share),
        "code6_share": float(code6_share),
    }
    return insights


def save_json(metrics: Dict[str, object]) -> None:
    PROFILE_JSON.write_text(json.dumps(metrics, indent=2))


def run_pipeline(limit: int | None = None) -> None:
    raw_df = load_raw_data(limit)
    clean_df = clean_dataframe(raw_df)
    clean_df.to_parquet(INTERIM_PATH, index=False)
    featured_df = engineer_features(clean_df)
    featured_df.to_parquet(PROCESSED_PATH, index=False)

    agg_df = aggregate_area_hour(featured_df)
    agg_df.to_parquet(AGG_AREA_HOUR_PATH, index=False)
    aggregate_daily_counts(featured_df).to_parquet(DAILY_VOLUME_PATH, index=False)
    aggregate_call_types(featured_df).to_parquet(CALL_TYPE_SUMMARY_PATH, index=False)
    aggregate_monthly_priority(featured_df).to_parquet(MONTHLY_PRIORITY_PATH, index=False)
    aggregate_family_month(featured_df).to_parquet(CALL_FAMILY_MONTH_PATH, index=False)
    run_ml_lab(featured_df)

    figures = run_eda_outputs(featured_df)
    metrics = compute_quality_metrics(featured_df)
    insights = compute_insights(featured_df)
    schema_profile = summarize_schema(featured_df, SAMPLE_COLUMNS)
    categorical_overview = build_categorical_overview(featured_df)
    payload = metrics | {
        "figures": figures,
        "insights": insights,
        "schema": schema_profile,
        "categorical_overview": categorical_overview,
    }
    save_json(payload)
    summary = build_summary_markdown(metrics, insights)
    SUMMARY_PATH.write_text(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LAPD Calls inspection and EDA pipeline.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for debugging.",
    )
    args = parser.parse_args()
    run_pipeline(limit=args.limit)


if __name__ == "__main__":
    main()
