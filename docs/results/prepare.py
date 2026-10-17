#!/usr/bin/env python3
"""Render the README ECE figure, optionally refreshing from the latest paper."""

import argparse
import hashlib
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


HERE = Path(__file__).resolve().parent
MODALITIES = ("text", "image", "video")
METHODS = {
    "sft_softlabel_2b": "Soft-label SFT (2B)",
    "cora_trained_2b": "CORA (2B)",
    "sft_softlabel_9b": "Soft-label SFT (9B)",
    "cora_trained_9b": "CORA (9B)",
    "shaped_rlcd_2b": "Shaped RLCD",
    "grpo_2b": "GRPO",
    "ctr_2b": "CtR (OpenJev-RLCD)",
    "rlcd_lite_2b": "RLCD-lite",
    "rewarding_doubt_2b": "Rewarding Doubt",
}
ABLATIONS = {
    r"\Cora{} (10 bins, group 256, support 20)": "CORA (default)",
    r"Bin accuracy $\to$ bin mean confidence": "Replace bin accuracy with bin mean confidence",
    r"Confidence bins $\to$ one global target": "Replace confidence bins with one global target",
    r"Trainable scope $\to$ head only": "Update the decision head only",
    r"Minimum support $20 \to 1$": "Reduce minimum bin support from 20 to 1",
}


def parse_metrics(cells, fields):
    values = [float(cell.strip().removesuffix(r"\\").strip()) for cell in cells]
    if len(values) != len(MODALITIES) * len(fields):
        raise ValueError(f"Unexpected metric count: {len(values)}")
    return {
        modality: dict(zip(fields, values[index * len(fields):(index + 1) * len(fields)]))
        for index, modality in enumerate(MODALITIES)
    }


def export_snapshot(paper_dir):
    source_paths = (
        "tables/main_results_position.tex",
        "tables/cora_ablation_results.tex",
        "sections/4.experiments.tex",
    )
    contents = {name: (paper_dir / name).read_bytes() for name in source_paths}
    rows = dict(re.findall(
        r"^% method-id: (\S+)\n([^\n]+)",
        contents[source_paths[0]].decode("utf-8"),
        re.MULTILINE,
    ))
    methods = []
    for identifier, name in METHODS.items():
        cells = rows[identifier].split("&")
        methods.append({
            "id": identifier,
            "name": name,
            "parameters_billion": int(cells[1].strip()),
            "metrics": parse_metrics(cells[2:], ("accuracy_pct", "confidence", "ece", "nll", "brier")),
        })

    ablation_rows = {
        cells[0].strip(): cells[1:]
        for line in contents[source_paths[1]].decode("utf-8").splitlines()
        if "&" in line
        for cells in [line.split("&")]
    }
    ablations = [{
        "name": name,
        "metrics": parse_metrics(ablation_rows[label], ("accuracy_pct", "confidence", "ece")),
    } for label, name in ABLATIONS.items()]
    default = next(row for row in methods if row["id"] == "cora_trained_2b")["metrics"]
    if any(default[modality][metric] != value
           for modality, metrics in ablations[0]["metrics"].items()
           for metric, value in metrics.items()):
        raise ValueError("Main-table and ablation default metrics disagree")

    return {
        "description": "Reported rounded values from the latest manuscript; no new inference.",
        "paper_title": "UniJev: Separating Accuracy from Calibration in Trimodal System One Decision Models",
        "sources": [{"path": f"newpaper/{name}", "sha256": hashlib.sha256(data).hexdigest()}
                    for name, data in contents.items()],
        "evaluation": {
            "questions": {"text": 764, "image": 1858, "video": 502},
            "ece_bins": 15,
            "ece_binning": "equal-width",
            "accuracy_unit": "percent",
        },
        "training": {
            "cora_updates": 100,
            "local_rlcd_step_budget": 500,
            "ablation_updates": 100,
            "default_cora": {"bins": 10, "group_questions": 256, "minimum_bin_support": 20},
        },
        "methods": methods,
        "ablations": ablations,
    }


def plot(snapshot):
    methods = {row["id"]: row["metrics"] for row in snapshot["methods"]}
    colors = {"sft_softlabel": "#BBC4CF", "cora_trained": "#187768"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, facecolor="white")
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.13, top=0.70, wspace=0.16)
    fig.text(0.075, 0.94, "CORA improves visual confidence calibration",
             fontsize=19, fontweight="bold", color="#183D37")
    total = sum(snapshot["evaluation"]["questions"].values())
    updates = snapshot["training"]["cora_updates"]
    fig.text(0.075, 0.875, f"{total:,} held-out questions  ·  {updates} CORA updates  ·  Lower ECE is better",
             fontsize=12, color="#54635F")
    fig.legend(handles=[Patch(color=colors[key], label=label) for key, label in
                        (("sft_softlabel", "Soft-label SFT"), ("cora_trained", "CORA"))],
               loc="upper center", bbox_to_anchor=(0.53, 0.85), ncol=2, frameon=False, fontsize=12)

    max_ece = max(row[modality]["ece"] for key, row in methods.items()
                  if key.startswith(("sft_softlabel", "cora_trained")) for modality in MODALITIES)
    upper = max(0.10, max_ece * 1.3)
    for ax, scale in zip(axes, (2, 9)):
        for offset, method in ((-0.18, "sft_softlabel"), (0.18, "cora_trained")):
            values = [methods[f"{method}_{scale}b"][modality]["ece"] for modality in MODALITIES]
            bars = ax.bar([index + offset for index in range(3)], values, width=0.32,
                          color=colors[method], zorder=3)
            ax.bar_label(bars, fmt="%.3f", padding=5, fontsize=12, color="#233A35")
        ax.set_title(f"{scale}B model", loc="left", fontsize=15, fontweight="bold", pad=10)
        ax.set_xticks(range(3), [name.title() for name in MODALITIES])
        ax.set_ylim(0, upper)
        ax.grid(axis="y", color="#E7ECE9", linewidth=0.8, zorder=0)
        ax.tick_params(axis="both", length=0, pad=9, labelcolor="#54635F")
        for spine in ("top", "left", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#D5DED9")
    axes[0].set_ylabel(f'ECE ({snapshot["evaluation"]["ece_bins"]} bins)', labelpad=10, color="#54635F")
    fig.savefig(HERE / "calibration.png", dpi=180, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, help="Refresh metrics from this manuscript directory")
    args = parser.parse_args()
    snapshot_path = HERE / "results.json"
    if args.paper_dir:
        snapshot = export_snapshot(args.paper_dir)
        snapshot_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    plot(snapshot)
    print(f"Rendered {HERE / 'calibration.png'}")


if __name__ == "__main__":
    main()
