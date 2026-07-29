#!/usr/bin/env python
"""Generate a report markdown file from a metrics.json, following the shared
8-section template. Quantitative sections (1, 2 partially, 3, 4, 5) are
filled in automatically wherever the data is available; qualitative
sections (6, 7, 8, and any metadata not passed on the CLI) are left as
placeholders for a person to fill in by hand.

Usage:
    python scripts/make_report.py results/majority_class__dummy__B1.metrics.json \
        --output reports/majority_class__dummy__B1.md \
        --tester "<role>" --setup-location "local" \
        --baseline results/rule_based__dummy__B1.metrics.json \
        --leak results/majority_class__dummy__B1.leak.json \
        --shortcut results/majority_class__dummy__B1.shortcut.json
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

PLACEHOLDER = "_TODO: fill in by hand_"


def _load_json(path: str | None) -> dict | None:
    if path is None:
        return None
    with open(path) as f:
        return json.load(f)


def render_report(
    metrics: dict,
    baseline_metrics: dict | None,
    leak: dict | None,
    shortcut: dict | None,
    tester: str,
    setup_location: str,
    setup_cost: str,
    trained: bool,
) -> str:
    model = metrics.get("model", PLACEHOLDER)
    dataset = metrics.get("dataset", PLACEHOLDER)
    task = metrics.get("task", PLACEHOLDER)
    metric_values = metrics.get("metrics", {})

    lines: list[str] = []
    lines.append(f"# Report: {model} on {dataset} ({task})")
    lines.append("")

    lines.append("## 1. Basic info")
    lines.append(f"- Model: {model}")
    lines.append(f"- Dataset: {dataset}")
    lines.append(f"- Task: {task}")
    lines.append(f"- Date: {date.today().isoformat()}")
    lines.append(f"- Tester (role, not name): {tester}")
    lines.append("")

    lines.append("## 2. Setup")
    lines.append(f"- Run location / cost: {setup_location} / {setup_cost}")
    lines.append(f"- Trained: {'yes (has fit() step)' if trained else 'no (zero-shot)'}")
    lines.append(f"- Model version/checkpoint: {PLACEHOLDER}")
    lines.append("")

    lines.append("## 3. Leak check")
    if leak is not None:
        lines.append(f"- Threshold: {leak.get('threshold')}")
        lines.append(f"- Test samples flagged: {leak.get('n_flagged')} / {leak.get('n_test')}")
        lines.append(f"- Max train/test similarity: {leak.get('max_similarity')}")
        conclusion = "independent enough to trust" if not leak.get("n_flagged") else "NOT independent - investigate before trusting Section 4"
        lines.append(f"- Conclusion: {conclusion}")
    else:
        lines.append(f"- {PLACEHOLDER} (run scoring/leak_check.py and pass --leak)")
    lines.append("")

    lines.append("## 4. Main results")
    lines.append(f"- n_predictions: {metrics.get('n_predictions')}")
    lines.append("")
    lines.append("| metric | this model | baseline | delta |")
    lines.append("|---|---|---|---|")
    baseline_values = (baseline_metrics or {}).get("metrics", {})
    for metric_name, value in metric_values.items():
        baseline_value = baseline_values.get(metric_name)
        if baseline_value is not None and isinstance(value, (int, float)) and isinstance(baseline_value, (int, float)):
            delta = value - baseline_value
            lines.append(f"| {metric_name} | {value} | {baseline_value} | {delta:+.4f} |")
        else:
            lines.append(f"| {metric_name} | {value} | {baseline_value if baseline_value is not None else PLACEHOLDER} | {PLACEHOLDER} |")
    lines.append("")

    lines.append("## 5. Shortcut check (video/image only)")
    if shortcut is not None:
        lines.append(f"- Real-input score: {shortcut.get('real_score')}")
        lines.append(f"- Static-frame score: {shortcut.get('static_frame_score')}")
        lines.append(f"- Relative drop: {shortcut.get('relative_drop')}")
        lines.append(f"- Verdict: {shortcut.get('verdict')}")
    else:
        lines.append(f"- {PLACEHOLDER} (not applicable for numeric/audio datasets; "
                      "otherwise run scoring/shortcut_check.py and pass --shortcut)")
    lines.append("")

    lines.append("## 6. Specific error examples")
    lines.append(f"{PLACEHOLDER} - list at least 3-5 wrong predictions from the test split with: "
                  "(1) predicted, (2) ground truth, (3) likely cause "
                  "(missing input / preprocessing artifact / hallucination / other).")
    lines.append("")

    lines.append("## 7. Conclusion")
    lines.append("- [ ] This model WINS over the simple baseline on this dataset")
    lines.append("- [ ] This model TIES the simple baseline (not worth using separately here)")
    lines.append("- [ ] This model LOSES to the simple baseline on this dataset")
    lines.append("")

    lines.append("## 8. Notes / limitations")
    lines.append(f"- Sample size adequate? {PLACEHOLDER}")
    lines.append(f"- Leak detected? Does it undermine Section 4? {PLACEHOLDER}")
    lines.append(f"- What would make this result more trustworthy? {PLACEHOLDER}")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=None, help="another metrics.json to compare against")
    parser.add_argument("--leak", type=Path, default=None, help="leak_check.py output json")
    parser.add_argument("--shortcut", type=Path, default=None, help="shortcut_check.py output json")
    parser.add_argument("--tester", default=PLACEHOLDER, help="role, not a person's name")
    parser.add_argument("--setup-location", default=PLACEHOLDER)
    parser.add_argument("--setup-cost", default=PLACEHOLDER)
    parser.add_argument("--trained", action="store_true", help="set if the model has a fit() step (not zero-shot)")
    args = parser.parse_args()

    metrics = _load_json(str(args.metrics_path))
    baseline_metrics = _load_json(str(args.baseline)) if args.baseline else None
    leak = _load_json(str(args.leak)) if args.leak else None
    shortcut = _load_json(str(args.shortcut)) if args.shortcut else None

    report = render_report(
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        leak=leak,
        shortcut=shortcut,
        tester=args.tester,
        setup_location=args.setup_location,
        setup_cost=args.setup_cost,
        trained=args.trained,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
