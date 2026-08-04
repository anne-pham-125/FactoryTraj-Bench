# Report: majority_class on dummy (B1)

## 1. Basic info
- Model: majority_class
- Dataset: dummy
- Task: B1
- Date: 2026-07-29
- Tester (role, not name): smoke-test

## 2. Setup
- Run location / cost: local / _TODO: fill in by hand_
- Trained: no (zero-shot)
- Model version/checkpoint: _TODO: fill in by hand_

## 3. Leak check
- _TODO: fill in by hand_ (run scoring/leak_check.py and pass --leak)

## 4. Main results
- n_predictions: 12

| metric | this model | baseline | delta |
|---|---|---|---|
| macro_f1 | 0.16666666666666666 | _TODO: fill in by hand_ | _TODO: fill in by hand_ |

## 5. Shortcut check (video/image only)
- _TODO: fill in by hand_ (not applicable for numeric/audio datasets; otherwise run scoring/shortcut_check.py and pass --shortcut)

## 6. Specific error examples
_TODO: fill in by hand_ - list at least 3-5 wrong predictions from the test split with: (1) predicted, (2) ground truth, (3) likely cause (missing input / preprocessing artifact / hallucination / other).

## 7. Conclusion
- [ ] This model WINS over the simple baseline on this dataset
- [ ] This model TIES the simple baseline (not worth using separately here)
- [ ] This model LOSES to the simple baseline on this dataset

## 8. Notes / limitations
- Sample size adequate? _TODO: fill in by hand_
- Leak detected? Does it undermine Section 4? _TODO: fill in by hand_
- What would make this result more trustworthy? _TODO: fill in by hand_
