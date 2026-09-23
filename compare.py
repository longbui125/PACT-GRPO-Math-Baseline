"""Collect available evaluations and training diagnostics into one table."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pact_grpo.comparison import build_comparison

SCALE = "verified_pilot"  # Change to pilot or main for other runs.

if __name__ == "__main__":
    csv_path, markdown_path = build_comparison(ROOT / "outputs" / SCALE)
    print(csv_path)
    print(markdown_path)
