from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError, CommandParser

from insights.ml_lab import LabConfig, generate_lab_report

SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".parquet"}


def _load_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise CommandError(f"Unsupported file extension '{suffix}'.")
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_parquet(path)


class Command(BaseCommand):
    help = "Run the ML Lab: split, train, evaluate, and write reports/ml_lab.json"

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument(
            "--data",
            type=str,
            default=str(settings.ML_LAB["data_path"]),
            help="Path to the labeled dataset (csv, tsv, parquet).",
        )
        parser.add_argument(
            "--target",
            type=str,
            default=settings.ML_LAB["target"],
            help="Name of the target column.",
        )
        parser.add_argument(
            "--synthetic",
            type=str,
            choices=["auto", "off", "smote"],
            default=settings.ML_LAB.get("synthetic", "auto"),
            help="Synthetic sampling policy. 'auto' enables SMOTE when imbalance is detected.",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=int(settings.ML_LAB.get("seed", 13)),
            help="Random seed for deterministic splits.",
        )
        parser.add_argument(
            "--report",
            type=str,
            default=str(settings.ML_LAB["report_path"]),
            help="Destination path for the JSON report.",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        data_path = Path(options["data"]).expanduser().resolve()
        if not data_path.exists():
            raise CommandError(f"Data file not found: {data_path}")

        target = options["target"]
        synthetic = options["synthetic"]
        seed = options["seed"]
        report_path = Path(options["report"]).expanduser().resolve()

        self.stdout.write(f"Loading dataset from {data_path} ...")
        frame = _load_frame(data_path)
        if target not in frame.columns:
            raise CommandError(f"Target column '{target}' is not present in {data_path}.")

        config = LabConfig(
            target=target,
            report_path=report_path,
            synthetic=synthetic,
            seed=seed,
            rf_params=settings.ML_LAB.get("rf_params"),
            max_rows=settings.ML_LAB.get("max_rows"),
        )

        report = generate_lab_report(frame, target, config)
        metrics: Dict[str, Any] = report.get("metrics", {}).get("test", {})
        accuracy = metrics.get("accuracy")
        f1 = metrics.get("f1_macro")
        accuracy_display = f"{accuracy:.3f}" if accuracy is not None else "n/a"
        f1_display = f"{f1:.3f}" if f1 is not None else "n/a"
        self.stdout.write(
            self.style.SUCCESS(
                f"ML Lab complete: test accuracy={accuracy_display} | macro F1={f1_display}".rstrip(".")
            )
        )
        self.stdout.write(self.style.SUCCESS(f"Report -> {report_path}"))
