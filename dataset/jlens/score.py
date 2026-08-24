"""Score collected_introspection.jsonl: overlap@10 / precision / recall / F1 by condition.

    python score.py
"""

from __future__ import annotations

import json
from collections import defaultdict

import config
from aggregate import normalize_token


def _read_jsonl(path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _row_metrics(row: dict) -> dict[str, float]:
    target = {normalize_token(item["concept"]) for item in row["j_lens_top10"]}
    predicted = {normalize_token(token) for token in row["predicted_top10"]}
    intersection = target & predicted

    overlap_at_10 = len(intersection) / 10
    precision = len(intersection) / len(predicted) if predicted else 0.0
    recall = len(intersection) / len(target) if target else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"overlap_at_10": overlap_at_10, "precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    rows = _read_jsonl(config.COLLECTED_INTROSPECTION_PATH)
    if not rows:
        print(f"No rows in {config.COLLECTED_INTROSPECTION_PATH}")
        return

    by_condition: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_condition[row["introspection_condition"]].append(row)

    header = f"{'condition':<20}{'n':>5}{'overlap@10':>12}{'precision':>12}{'recall':>10}{'f1':>10}{'valid<10':>10}"
    print(header)
    print("-" * len(header))
    for condition in config.INTROSPECTION_CONDITIONS:
        condition_rows = by_condition.get(condition, [])
        if not condition_rows:
            continue
        metrics = [_row_metrics(row) for row in condition_rows]
        n = len(metrics)
        under_10 = sum(1 for row in condition_rows if row["predicted_valid_count"] < 10)
        avg = {key: sum(m[key] for m in metrics) / n for key in metrics[0]}
        print(
            f"{condition:<20}{n:>5}{avg['overlap_at_10']:>12.3f}{avg['precision']:>12.3f}"
            f"{avg['recall']:>10.3f}{avg['f1']:>10.3f}{under_10:>10}"
        )


if __name__ == "__main__":
    main()
