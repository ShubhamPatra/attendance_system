"""Calibrate PPE thresholds from labeled confidence samples.

Input CSV must contain:
- label: one of none, mask, cap, both
- confidence: predicted confidence in [0, 1]

Example:
    python scripts/calibrate_ppe_threshold.py --csv data/ppe_scores.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass


VALID_LABELS = {"none", "mask", "cap", "both"}


@dataclass
class EvalPoint:
    threshold: float
    precision_pct: float
    recall_pct: float
    accuracy_pct: float


def load_samples(csv_path: str) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            label = str(row["label"]).strip().lower()
            if label not in VALID_LABELS:
                raise ValueError(f"Invalid PPE label: {label}")
            conf = float(row["confidence"])
            rows.append((label, conf))
    if not rows:
        raise ValueError("No rows found in CSV.")
    return rows


def evaluate_threshold(rows: list[tuple[str, float]], threshold: float) -> EvalPoint:
    tp = fp = fn = tn = 0
    for label, conf in rows:
        is_ppe = label in {"mask", "cap", "both"}
        predicted_ppe = conf >= threshold
        if is_ppe and predicted_ppe:
            tp += 1
        elif not is_ppe and predicted_ppe:
            fp += 1
        elif is_ppe and not predicted_ppe:
            fn += 1
        else:
            tn += 1

    total = tp + fp + fn + tn
    precision = (tp / (tp + fp) * 100.0) if (tp + fp) else 0.0
    recall = (tp / (tp + fn) * 100.0) if (tp + fn) else 0.0
    acc = ((tp + tn) / total * 100.0) if total else 0.0
    return EvalPoint(
        threshold=threshold,
        precision_pct=precision,
        recall_pct=recall,
        accuracy_pct=acc,
    )


def calibrate(rows: list[tuple[str, float]]) -> tuple[EvalPoint, EvalPoint]:
    points: list[EvalPoint] = []
    for step in range(1, 100):
        thr = step / 100.0
        points.append(evaluate_threshold(rows, thr))

    best_accuracy = max(points, key=lambda p: p.accuracy_pct)
    best_precision = max(points, key=lambda p: (p.precision_pct, p.recall_pct, p.accuracy_pct))
    return best_accuracy, best_precision


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate PPE confidence threshold.")
    parser.add_argument("--csv", required=True, help="Path to labeled CSV.")
    args = parser.parse_args()

    rows = load_samples(args.csv)
    best_accuracy, best_precision = calibrate(rows)

    print("Calibration summary")
    print("-------------------")
    print(
        f"Best accuracy threshold: {best_accuracy.threshold:.2f} "
        f"(ACC={best_accuracy.accuracy_pct:.2f}%, PREC={best_accuracy.precision_pct:.2f}%, "
        f"REC={best_accuracy.recall_pct:.2f}%)"
    )
    print(
        f"Best precision threshold: {best_precision.threshold:.2f} "
        f"(ACC={best_precision.accuracy_pct:.2f}%, PREC={best_precision.precision_pct:.2f}%, "
        f"REC={best_precision.recall_pct:.2f}%)"
    )


if __name__ == "__main__":
    main()
