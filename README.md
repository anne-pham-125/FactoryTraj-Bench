# FactoryTraj-Bench

![arXiv](https://img.shields.io/badge/arXiv-coming_soon-lightgrey)
![Dataset](https://img.shields.io/badge/datasets-7_sources-blue)
![Models](https://img.shields.io/badge/models-5_groups-green)
![License](https://img.shields.io/badge/license-TBD-lightgrey)

FactoryTraj-Bench measures whether AI models — multimodal, text-only, embedding-based, and classical baselines alike — genuinely understand and reason over real industrial data (assembly video, alarm/event logs, sensor trajectories, defect images, machine audio), or whether they are exploiting shortcuts (static appearance, class priors, train/test leakage) that happen to score well without any real comprehension of process state, causality, or recovery.

We deliberately avoid the term "world model": without action-conditioned outcome prediction (task B9) validated against a real or simulated environment, we are not yet in a position to claim models are learning a predictive world model of the factory floor — only that we can measure understanding and reasoning on frozen industrial trajectories.

## Our benchmark responds to the following questions

- Can a model recover the **schema** of an unfamiliar industrial signal from just tag names, samples, and partial documentation (B0)?
- Can it estimate the machine's **current state** from a short observation window, and localize **anomalies** and their **root cause** when something goes wrong (B1, B3, B4)?
- Can it predict what happens **next**, ground its actions in a written **SOP**, and rank candidate **recovery actions** (B5, B6, B8)?
- When we actively probe for shortcuts — feed it a single static frame instead of real video, check whether train and test samples are near-duplicates — does its score survive, or does it collapse?
- How much of the answer to all of the above comes from general multimodal pretraining (MLLM zero-shot) versus an embedding-native architecture (V-JEPA 2) versus a purpose-built classical model (XGBoost, Chronos-2) versus language ability alone (text-only LLM) versus the best commercial model on the market (GPT-4o, Gemini) — and where does each one actually fail?

## Leaderboard

_Empty until real runs land in `results/` and `reports/`. One row per (model, dataset, task)._

| Model | Group | Dataset | Task | Primary metric | Score | vs. baseline | Leak/shortcut checked? | Report |
|---|---|---|---|---|---|---|---|---|
| _TBD_ | | | | | | | | |

## Directory structure

```
FactoryTraj-Bench/
├── README.md
├── CONTRIBUTING.md
├── pyproject.toml
├── requirements.txt
├── .gitignore                  # data/, results/*.jsonl, checkpoints/, __pycache__, .env
├── .env.example                # API key placeholders (never commit real .env)
├── configs/
│   ├── tasks.yaml               # spec B0-B10: id, name, input, output, primary_metric, secondary_metrics
│   ├── models.yaml               # registry of 5 model groups: group, needs_training, research_question, models[]
│   └── compat_matrix.yaml        # allow-list of (dataset, model, task) triples - nothing runs if not declared here
├── data/
│   └── README.md                 # how to download each source (script or link); raw data is never committed
├── src/
│   └── worldbench/
│       ├── types.py              # Sample, Prediction, TaskSpec - the one shared schema
│       ├── runner.py             # shared run_eval() used by scripts/run_eval.py and every evaluation/examples/*/run.py
│       ├── datasets/             # one module per source, each exposing load(split) -> list[Sample]
│       │   ├── tep.py / alpi.py / hatrec.py / assembly101.py / industreal.py / mimii.py / mmad.py
│       │   └── registry.py       # dataset_id -> loader module
│       ├── tasks/                # one module per task: build_prompt(), parse_output(), compute_metric()
│       │   ├── b0_schema_tag.py ... b10_ood_recognition.py
│       │   ├── metrics.py        # shared metric implementations
│       │   └── registry.py       # task_id -> TaskSpec + handler module
│       ├── models/
│       │   ├── base.py           # ModelWrapper interface: predict(), optional fit()
│       │   ├── <model_id>.py     # one concrete wrapper per model in configs/models.yaml
│       │   └── registry.py       # model_id -> ModelWrapper class
│       └── scoring/
│           ├── score_run.py      # shared entrypoint: predictions.jsonl -> metrics.json, validates compat_matrix
│           ├── leak_check.py     # embedding similarity train/test, flags if > 0.95
│           └── shortcut_check.py # static-frame vs real video/image comparison
├── scripts/
│   ├── run_eval.py                # CLI: --model X --dataset Y --task Z --split test_locked
│   └── make_report.py             # generates the 8-section report from a metrics.json
├── evaluation/examples/           # one folder per MODEL (never per person), runnable by hand
│   ├── cosmos3_nano/run.py
│   ├── qwen2vl/run.py
│   ├── vjepa2/{run.py, requirements.txt}
│   ├── xgboost_chronos/run.py
│   ├── llm_text_only/run.py
│   ├── commercial_api/{gpt4o.py, gemini.py, requirements.txt}
│   └── baselines_simple/run.py    # rule-based, majority-class/persistence
├── results/                       # raw output: {model}__{dataset}__{task}.jsonl + .metrics.json, not hand-edited
├── reports/                       # generated markdown reports, qualitative sections filled by hand
└── checkpoints/                   # trained baseline/probe weights, gitignored
```

No person names appear anywhere in this repository (files, folders, variables, comments, commit messages). Task assignment and timelines are internal planning documents kept entirely outside this codebase.

## Pipeline

1. **Data prep** — Each source under `src/worldbench/datasets/` exposes `load(split) -> list[Sample]`. Raw data is downloaded separately per `data/README.md` and never committed. A `test_split_locked` split, once frozen, is never regenerated or hand-edited (see CONTRIBUTING.md).
2. **Model config** — Each model is declared once in `configs/models.yaml` and implemented as a `ModelWrapper` under `src/worldbench/models/`, wired to a thin runnable script under `evaluation/examples/<model>/run.py`. `configs/compat_matrix.yaml` is the single source of truth for which (dataset, model, task) triples are valid.
3. **Run eval** — `scripts/run_eval.py --model X --dataset Y --task Z --split test_locked` validates the combination, runs the model, and writes `results/{model}__{dataset}__{task}.jsonl` plus a `.metrics.json` via `src/worldbench/scoring/score_run.py`. Every report (`scripts/make_report.py` scaffold or written by hand) follows the same 8 sections: (1) basic info, (2) setup, (3) leak check, (4) main results vs. baseline, (5) shortcut check, (6) specific error examples, (7) conclusion, (8) notes/limitations.

## Domain knowledge

Task B3 on the `mmad` source reuses [MMAD](https://github.com/jam-cc/MMAD) (ICLR 2025) as one of our 7 data sources, including their `domain_knowledge.json` for industrial defect context. This is **their** data and annotation work, not ours — credit MMAD's authors wherever these results are reported, and check their license before any commercial use (see `data/README.md`). MMAD also published baselines for GPT-4o, Gemini, Qwen2.5-VL, InternVL, LLaVA and a human baseline (GPT-4o ≈ 74.9%); where we run a model on the same `mmad` split, we compare directly against those published numbers rather than re-deriving our own baseline.

## Todo

- [ ] Real data downloaded and loaders implemented for all 7 sources (currently stubs, see `src/worldbench/datasets/`)
- [ ] Model inference wired up for all 5 groups (currently stubs except `majority_class`, see `src/worldbench/models/`)
- [ ] Ground-truth data sourced for B2, B4, B6, B7, B8, B9 (no dataset covers them yet, see `configs/compat_matrix.yaml`)
- [ ] `test_split_locked` frozen per dataset
- [ ] First real leaderboard entries
- [ ] License finalized for this repository

## Citation

_Pending paper. This section will be filled in once a citable reference exists._
