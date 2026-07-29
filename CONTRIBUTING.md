# Contributing

This is a short internal guide for the team running evaluations on this benchmark. It assumes you've cloned the repo and run `pip install -e .` (or set `PYTHONPATH=src`) plus `pip install -r requirements.txt`.

No person names anywhere in this repository — not in file names, folder names, variables, comments, or commit messages. If you need to note who ran something, put that in your own report's metadata field (a role, not a name) or in an internal doc outside this repo.

## Adding a new dataset loader

1. Add a row to `data/README.md` describing where to download it and its license.
2. Create `src/worldbench/datasets/<source>.py` implementing `load(split) -> list[Sample]` (see any existing module for the pattern: log a warning and return `[]` if the raw data isn't present locally, don't raise a bare `NotImplementedError`).
3. Register it in `src/worldbench/datasets/registry.py` (`DATASET_LOADERS`).
4. Add entries to `configs/compat_matrix.yaml` under `datasets.<source>.entries` for every (model, task) pair it actually supports. `score_run.py` rejects anything not listed here — this is the only place that decides what's allowed to run.

## Adding a new model wrapper

1. Add the model to the right group in `configs/models.yaml` (or a new group if it genuinely answers a new research question).
2. Create `src/worldbench/models/<model_id>.py` with a class implementing `ModelWrapper` (`src/worldbench/models/base.py`): `predict(sample, prompt)`, and `fit(train_samples)` if `needs_training` is true. Save any trained weights to `checkpoints/` (gitignored).
3. Register it in `src/worldbench/models/registry.py` (`MODEL_CLASSES`).
4. Add a thin runnable script under `evaluation/examples/<model_id>/run.py` (or add it to an existing shared folder like `baselines_simple/` or `commercial_api/` if it's a natural fit) — it should be runnable by hand with just `--dataset`/`--task`/`--split` flags, calling `worldbench.runner.run_eval()`.
5. Add its (dataset, task) entries to `configs/compat_matrix.yaml` — same rule as above, nothing runs if it isn't declared.

## Running an evaluation

```
python scripts/run_eval.py --model <model_id> --dataset <dataset_id> --task <task_id> --split test_locked
```

or run the model's own script directly, e.g. `python evaluation/examples/baselines_simple/run.py --model majority_class --dataset <dataset_id> --task <task_id>`. Either path writes `results/{model}__{dataset}__{task}.jsonl` and `.metrics.json`.

`scripts/make_report.py` is an **optional** convenience that scaffolds a report from that output:

```
python scripts/make_report.py results/<model>__<dataset>__<task>.metrics.json --output reports/<model>__<dataset>__<task>.md
```

It auto-fills the model/dataset/task names and this model's score into the 8-section structure; every qualitative section (error examples, conclusion, notes) is still a `_TODO: fill in by hand_` placeholder either way. If you'd rather just write the report from scratch, that's fine too — as long as it follows the same 8 sections (see README.md's report template) so reports stay comparable across the team.

## `test_split_locked` is frozen once created

Once a dataset's `test_split_locked` split has been generated and used for even one real run, it must not be regenerated, reshuffled, or hand-edited. Changing it after the fact invalidates every existing result on that split and silently breaks comparability between reports. If you find a genuine problem with a locked split (mislabeled sample, corrupted file), open a PR that documents the exact problem and the fix — don't just quietly regenerate it.

## Submitting results

1. Open a PR that adds your `results/{model}__{dataset}__{task}.jsonl` + `.metrics.json` and your filled-in `reports/{model}__{dataset}__{task}.md`.
2. Run `src/worldbench/scoring/leak_check.py` and (for video/image tasks) `shortcut_check.py` before submitting, and include their output in the report's Sections 3 and 5.
3. If you're evaluating on a dataset someone else is already using, reuse their baseline numbers in your report's Section 4 instead of re-running the baseline yourself.
4. Keep the PR scoped to one (model, dataset) pair (or a small related batch) so it's easy to review against `configs/compat_matrix.yaml`.
