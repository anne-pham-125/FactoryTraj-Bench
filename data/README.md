# Data sources

Raw data is never committed to this repository. Each source below is
downloaded separately into `data/<source>/` (gitignored) by the person
running that dataset's loader.

| Source | Modality | Where to get it | License note |
|---|---|---|---|
| Tennessee Eastman Process (TEP) | Tabular / numeric | commonly mirrored on Kaggle / Harvard Dataverse - pick one mirror and record the exact URL used | Public / research use |
| ALPI / PIADE | Event / alarm log (tabular) | https://zenodo.org/records/7071747 | Check Zenodo record for license |
| MIMII | Audio | https://zenodo.org/record/3384388 (domain-shift variant: https://zenodo.org/records/4740355) | Check Zenodo record for license |
| MMAD | Static industrial images (8,366 images / 39,672 MCQ) | Repo: https://github.com/jam-cc/MMAD · Dataset: https://huggingface.co/datasets/jiang-cc/MMAD | Academic license from original authors - confirm before any commercial use. Credit MMAD (ICLR 2025) wherever results are reported. |
| HATREC | Video (assembly) | internal download - see team's shared data location, not a public URL | Confirm license before external release |
| Assembly101 | Video (assembly) | https://assembly-101.github.io/ | Check site for license; download the full dataset, not the sample subset (sample only has 1 real mistake event) |
| IndustReal | Video (assembly, procedural errors) | https://timschoonbeek.github.io/industreal | Check site for license |

## Adding a new source

1. Add a row to the table above with the download location and license note.
2. Write `src/worldbench/datasets/<source>.py` implementing `load(split) -> list[Sample]`.
3. Add the source to `configs/compat_matrix.yaml` with the (model, task) pairs it actually supports - nothing runs against a dataset that isn't declared there.
4. Update `data/README.md` again if the download step needs more than "go to this URL".
