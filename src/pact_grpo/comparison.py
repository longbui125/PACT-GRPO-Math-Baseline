from __future__ import annotations

import csv
import json
from pathlib import Path

METHODS = ["base", "grpo", "dr_grpo", "pcgrad_grpo", "pact_grpo"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def build_comparison(outputs_dir: str | Path) -> tuple[Path, Path]:
    root = Path(outputs_dir)
    rows = []
    for method in METHODS:
        evaluation = read_json(root / method / "evaluation" / "metrics.json")
        training = read_json(root / method / "summary.json")
        rows.append({
            "method": method,
            "gsm8k_accuracy": evaluation.get("gsm8k", {}).get("accuracy"),
            "math500_accuracy": evaluation.get("math500", {}).get("accuracy"),
            "mean_training_reward": training.get("mean_training_reward"),
            "mean_post_update_rsvr": training.get("mean_post_update_rsvr"),
            "mean_projection_norm_ratio": training.get("mean_projection_norm_ratio"),
            "projection_step_rate": training.get("projection_step_rate"),
            "training_seconds": training.get("elapsed_seconds"),
        })
    csv_path = root / "comparison.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def fmt(value, percent=False):
        if value is None:
            return ""
        return f"{100 * value:.2f}%" if percent else f"{value:.4f}"

    lines = [
        "# Preliminary benchmark comparison", "",
        "| Method | GSM8K | MATH-500 | Mean reward | RSVR | Update ratio | Projection rate | Seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {fmt(row['gsm8k_accuracy'], True)} | {fmt(row['math500_accuracy'], True)} | "
            f"{fmt(row['mean_training_reward'])} | {fmt(row['mean_post_update_rsvr'], True)} | "
            f"{fmt(row['mean_projection_norm_ratio'])} | {fmt(row['projection_step_rate'], True)} | "
            f"{fmt(row['training_seconds'])} |"
        )
    lines += ["", "Published paper scores are excluded because this table is a controlled local comparison under one shared setup."]
    markdown_path = root / "comparison.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, markdown_path

