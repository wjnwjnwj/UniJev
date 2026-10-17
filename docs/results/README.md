# Paper results in the README

The results section uses the latest manuscript, **UniJev: Separating Accuracy from Calibration in Trimodal System One Decision Models**, from `newpaper/` in the research workspace.

[results.json](results.json) contains the selected main-table and ablation rows, evaluation settings, and SHA-256 hashes of the source files:

- `newpaper/tables/main_results_position.tex`: soft-label SFT and CORA at 2B/9B, plus the five local 2B RLCD continuations.
- `newpaper/tables/cora_ablation_results.tex`: the default and four single-factor variants shown in the README.
- `newpaper/sections/4.experiments.tex`: evaluation protocol and interpretation.

The snapshot preserves the tables' reported precision: accuracy in percent and ECE, NLL, Brier, and mean confidence on their original scales. Relative ECE reductions in the README are calculated from those rounded values as `(SFT ECE − CORA ECE) / SFT ECE`. They are point estimates, with no uncertainty intervals added.

[calibration.png](calibration.png) plots the ECE values from the snapshot. Both panels use the same zero-based vertical scale. The figure includes all three modalities at both model sizes, including the increase in 9B text ECE. The source snapshot also retains the other main-table metrics; for example, 2B text Brier increases from 0.387 to 0.400 even though its NLL decreases.

Evaluation uses 15 equal-width bins on 3,124 test questions. Default CORA training uses 10 bins, groups of 256 questions, minimum bin support 20, and 100 updates. The local RLCD comparisons use a 500-step budget from the same 2B initialization; they are not a matched-budget comparison. All displayed ablation arms use the same 100-update protocol.

## Rebuild the figure

With Python 3 and Matplotlib installed, run from the repository root:

```bash
python docs/results/prepare.py
```

This reads the bundled snapshot and regenerates the figure. The repository is self-contained for readers and does not need access to the research workspace.

To refresh the snapshot from the paper's source tables and regenerate the figure:

```bash
python docs/results/prepare.py --paper-dir /path/to/newpaper
```

The exporter selects rows by their method identifiers or ablation labels and does not run inference. After refreshing, review the evaluation settings against the manuscript and update the README's tables and interpretation to match the snapshot.
