"""VS Code entry point: report adaptation, retention, and training cost."""

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OPTIONS = json.loads((ROOT / "configs" / "transfer.json").read_text(encoding="utf-8"))
OUTPUT = ROOT / "outputs" / "transfer_v1"


def read_run(base: Path, method: str, split: str) -> tuple[dict, list[dict]]:
    folder = base / method / split
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    predictions = [json.loads(line) for line in (folder / "predictions.jsonl").read_text(
        encoding="utf-8").splitlines()]
    if metrics["samples"] != len(predictions):
        raise ValueError(f"Incomplete evaluation: {folder}")
    return metrics, predictions


def paired(left: list[dict], right: list[dict]) -> dict:
    if [r["id"] for r in left] != [r["id"] for r in right]:
        raise ValueError("Paired results use different examples or order")
    return {
        "right_only_correct": sum(not a["correct"] and b["correct"]
                                  for a, b in zip(left, right)),
        "left_only_correct": sum(a["correct"] and not b["correct"]
                                 for a, b in zip(left, right)),
    }


def validation_curve(base: Path, method: str) -> list[dict]:
    if method == "initial":
        checkpoints = [(0, base / "initial")]
        logs = []
    else:
        summary = json.loads((base / method / "summary.json").read_text(encoding="utf-8"))
        checkpoints = [(step, base / method / "checkpoints" / f"step_{step}")
                       for step in summary["checkpoints"] if step != summary["target_groups"]]
        checkpoints.append((summary["target_groups"], base / method))
        logs = [json.loads(line) for line in (base / method / "train_metrics.jsonl").read_text(
            encoding="utf-8").splitlines()]
    curve = []
    for step, folder in checkpoints:
        easy = json.loads((folder / "validation_easy_zero" / "metrics.json").read_text())
        hard = json.loads((folder / "validation_hard" / "metrics.json").read_text())
        eligible = [row for row in logs if row["step"] <= step]
        tokens = eligible[-1]["rollout_tokens"] if eligible else 0
        curve.append({"step": step, "rollout_tokens": tokens,
                      "easy_zero_rejection": easy["exact_set_accuracy"],
                      "hard_accuracy": hard["exact_set_accuracy"]})
    return curve


def summarize_seed(seed: int) -> tuple[list[dict], dict]:
    base = OUTPUT / f"seed_{seed}"
    methods = ("initial", *OPTIONS["methods"])
    predictions = {}
    rows = []
    curves = {}
    for method in methods:
        easy_metrics, easy_pred = read_run(base, method, "test_easy")
        hard_metrics, hard_pred = read_run(base, method, "test_hard")
        predictions[method] = {"easy": easy_pred, "hard": hard_pred}
        easy_zero = easy_metrics["by_root_count"]["0"]
        easy_one = easy_metrics["by_root_count"]["1"]
        easy_two = easy_metrics["by_root_count"]["2"]
        summary_file = base / ("initial_adapter" if method == "initial" else method) / "summary.json"
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        row = {
            "seed": seed, "method": method,
            "hard_correct": hard_metrics["correct"], "hard_total": hard_metrics["samples"],
            "hard_accuracy": hard_metrics["exact_set_accuracy"],
            "easy_zero_correct": easy_zero["correct"], "easy_zero_total": easy_zero["samples"],
            "easy_zero_rejection": easy_zero["correct"] / easy_zero["samples"],
            "easy_one_accuracy": easy_one["correct"] / easy_one["samples"],
            "easy_two_accuracy": easy_two["correct"] / easy_two["samples"],
            "easy_zero_false_acceptance": 1 - easy_zero["correct"] / easy_zero["samples"],
            "train_seconds": summary.get("elapsed_seconds"),
            "rollout_tokens": summary.get("rollout_tokens"),
            "sampled_groups": summary.get("sampled_groups"),
            "optimizer_updates": summary.get("optimizer_updates"),
            "projected_steps": summary.get("projected_steps"),
        }
        rows.append(row)
        curves[method] = validation_curve(base, method)
    initial = rows[0]
    for row in rows:
        row["hard_gain_pp"] = 100 * (row["hard_accuracy"] - initial["hard_accuracy"])
        row["easy_zero_change_pp"] = 100 * (row["easy_zero_rejection"]
                                            - initial["easy_zero_rejection"])
        row["adaptation_with_forgetting"] = (row["hard_gain_pp"] > 0
                                            and row["easy_zero_change_pp"] < 0)
    paired_results = {}
    for method in OPTIONS["methods"]:
        paired_results[method] = {
            split: paired(predictions["initial"][split], predictions[method][split])
            for split in ("easy", "hard")
        }
        paired_results[method]["easy_zero"] = paired(
            [row for row in predictions["initial"]["easy"] if len(row["gold"]) == 0],
            [row for row in predictions[method]["easy"] if len(row["gold"]) == 0],
        )
    for method in OPTIONS["methods"]:
        if method != "pact_grpo" and "pact_grpo" in predictions:
            paired_results[f"pact_vs_{method}"] = {
                split: paired(predictions[method][split],
                              predictions["pact_grpo"][split])
                for split in ("easy", "hard")
            }
            paired_results[f"pact_vs_{method}"]["easy_zero"] = paired(
                [row for row in predictions[method]["easy"] if len(row["gold"]) == 0],
                [row for row in predictions["pact_grpo"]["easy"] if len(row["gold"]) == 0],
            )
    return rows, {"validation_curves": curves, "paired": paired_results}


if __name__ == "__main__":
    all_rows = []
    details = {}
    for seed in OPTIONS["seeds"]:
        rows, detail = summarize_seed(int(seed))
        all_rows.extend(rows)
        details[str(seed)] = detail
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    aggregate = {}
    for method in ("initial", *OPTIONS["methods"]):
        selected = [row for row in all_rows if row["method"] == method]
        aggregate[method] = {name: statistics.mean(row[name] for row in selected)
                             for name in ("hard_accuracy", "easy_zero_rejection",
                                          "hard_gain_pp", "easy_zero_change_pp")}
    (OUTPUT / "comparison.json").write_text(json.dumps({
        "rows": all_rows, "mean_by_method": aggregate, "per_seed": details,
        "interpretation": "A hard gain with old-skill loss is measured, never assumed.",
    }, indent=2), encoding="utf-8")
    for row in all_rows:
        print(f"seed={row['seed']} {row['method']:10s} "
              f"hard={row['hard_accuracy']:.1%} "
              f"old_zero_rejection={row['easy_zero_rejection']:.1%} "
              f"tokens={row['rollout_tokens']}")
