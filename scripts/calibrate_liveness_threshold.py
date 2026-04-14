"""Calibrate liveness threshold from labeled confidence samples.

Input CSV must contain:
- is_real: 1 for genuine, 0 for spoof
- confidence: anti-spoof confidence score in [0, 1]

Example:
    python scripts/calibrate_liveness_threshold.py --csv data/liveness_scores.csv --far-target 1.0
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass


@dataclass
class EvalPoint:
    threshold: float
    far_pct: float
    frr_pct: float
    accuracy_pct: float


def load_samples(csv_path: str) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            label = int(row["is_real"])
            conf = float(row["confidence"])
            rows.append((label, conf))
    if not rows:
        raise ValueError("No rows found in CSV.")
    return rows


def evaluate_threshold(rows: list[tuple[int, float]], threshold: float) -> EvalPoint:
    tp = fp = fn = tn = 0
    for is_real, conf in rows:
        predicted_real = conf >= threshold
        if is_real == 1 and predicted_real:
            tp += 1
        elif is_real == 0 and predicted_real:
            fp += 1
        elif is_real == 1 and not predicted_real:
            fn += 1
        else:
            tn += 1

    total = tp + fp + fn + tn
    far = (fp / (fp + tn) * 100.0) if (fp + tn) else 0.0
    frr = (fn / (fn + tp) * 100.0) if (fn + tp) else 0.0
    acc = ((tp + tn) / total * 100.0) if total else 0.0
    return EvalPoint(threshold=threshold, far_pct=far, frr_pct=frr, accuracy_pct=acc)


def calibrate(rows: list[tuple[int, float]], far_target: float) -> tuple[EvalPoint, EvalPoint]:
    points: list[EvalPoint] = []
    for step in range(1, 100):
        thr = step / 100.0
        points.append(evaluate_threshold(rows, thr))

    best_accuracy = max(points, key=lambda p: p.accuracy_pct)

    far_compliant = [p for p in points if p.far_pct <= far_target]
    if far_compliant:
        best_far = min(far_compliant, key=lambda p: (p.frr_pct, -p.accuracy_pct))
    else:
        best_far = min(points, key=lambda p: p.far_pct)

    return best_accuracy, best_far


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate liveness confidence threshold.")
    parser.add_argument("--csv", required=True, help="Path to labeled CSV.")
    parser.add_argument(
        "--far-target",
        type=float,
        default=1.0,
        help="Target FAR percentage (default: 1.0).",
    )
    args = parser.parse_args()

    rows = load_samples(args.csv)
    best_accuracy, best_far = calibrate(rows, args.far_target)

    print("Calibration summary")
    print("-------------------")
    print(
        f"Best accuracy threshold: {best_accuracy.threshold:.2f} "
        f"(ACC={best_accuracy.accuracy_pct:.2f}%, FAR={best_accuracy.far_pct:.2f}%, "
        f"FRR={best_accuracy.frr_pct:.2f}%)"
    )
    print(
        f"Best FAR-constrained threshold: {best_far.threshold:.2f} "
        f"(ACC={best_far.accuracy_pct:.2f}%, FAR={best_far.far_pct:.2f}%, "
        f"FRR={best_far.frr_pct:.2f}%)"
    )


if __name__ == "__main__":
    main()
