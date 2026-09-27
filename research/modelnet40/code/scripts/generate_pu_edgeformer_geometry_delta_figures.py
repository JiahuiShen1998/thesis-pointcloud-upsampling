#!/usr/bin/env python3
"""Generate Line B geometry delta figures including PU-EdgeFormer (methods only)."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT / "reports"
FIG_DIR = PROJECT / "figures/modelnet40/geometry_delta_methods_only_with_pu_edgeformer"
THESIS_CSV = REPORTS / "modelnet40_quality_metrics_thesis_table_with_pu_edgeformer.csv"

METHODS_ONLY = ["EAR", "PDANS", "PU-Net", "PU-GCN", "PU-EdgeFormer"]
LINEB_GROUPS = {
    "EAR": "Downsampled x4 + EAR",
    "PDANS": "Downsampled x4 + PDANS",
    "PU-Net": "Downsampled x4 + PU-Net",
    "PU-GCN": "Downsampled x4 + PU-GCN",
    "PU-EdgeFormer": "Downsampled x4 + PU-EdgeFormer",
}
METRICS = [
    ("delta_CD_vs_original", "Δ CD", "lineB_delta_cd_vs_original_baseline_methods_only_with_pu_edgeformer"),
    ("delta_HD_vs_original", "Δ HD", "lineB_delta_hd_vs_original_baseline_methods_only_with_pu_edgeformer"),
    ("delta_NUC_vs_original", "Δ NUC", "lineB_delta_nuc_vs_original_baseline_methods_only_with_pu_edgeformer"),
    ("delta_P2F_vs_original", "Δ exact P2F", "lineB_delta_exact_p2f_vs_original_baseline_methods_only_with_pu_edgeformer"),
]

CAPTION = (
    "The Original baseline is used as the zero reference and is not plotted as a method bar. "
    "Absolute metrics were computed with respect to the dense mesh surface reference."
)


def load_deltas() -> dict[str, dict[str, float]]:
    with open(THESIS_CSV, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_group = {r["group"]: r for r in rows}
    out = {}
    for label, group in LINEB_GROUPS.items():
        r = by_group[group]
        out[label] = {
            "delta_CD_vs_original": float(r["delta_CD_vs_original"]),
            "delta_HD_vs_original": float(r["delta_HD_vs_original"]),
            "delta_NUC_vs_original": float(r["delta_NUC_vs_original"]),
            "delta_P2F_vs_original": float(r["delta_P2F_vs_original"]),
        }
    return out


def plot_one(deltas: dict[str, dict[str, float]], metric_key: str, ylabel: str, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    vals = [deltas[m][metric_key] for m in METHODS_ONLY]
    x = np.arange(len(METHODS_ONLY))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)
    bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.6, width=0.72)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", label="Original baseline (y=0)")
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS_ONLY, rotation=15)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Line B {ylabel} vs Original baseline (methods only)")
    ymax = max(abs(v) for v in vals) if vals else 1.0
    pad = ymax * 0.18 + 1e-4
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + (pad * 0.15 if v >= 0 else -pad * 0.35),
            f"{v:+.4f}",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=8,
        )
    ax.set_ylim(min(0.0, min(vals)) - pad, max(0.0, max(vals)) + pad)
    ax.legend(loc="best", fontsize=8)
    fig.text(0.5, 0.01, CAPTION, ha="center", va="bottom", fontsize=7.5, wrap=True)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(FIG_DIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def update_docs() -> None:
    # Append/update figure index, captions, visualization summary with EdgeFormer notes.
    index = PROJECT / "figures/modelnet40/figure_index.md"
    captions = REPORTS / "modelnet40_thesis_figure_captions.md"
    summary = REPORTS / "modelnet40_visualization_summary.md"

    block_index = [
        "",
        "## Geometry delta methods-only (with PU-EdgeFormer)",
        "",
        "| Figure | Source | Caption | Section |",
        "|---|---|---|---|",
    ]
    for _, ylabel, stem in METRICS:
        block_index.append(
            f"| `geometry_delta_methods_only_with_pu_edgeformer/{stem}` | "
            f"`modelnet40_quality_metrics_thesis_table_with_pu_edgeformer.csv` | "
            f"Line B {ylabel} vs Original baseline (methods only, includes PU-EdgeFormer). | "
            f"Results — Geometry quality |"
        )
    block_index.append("")
    block_index.append(
        "- Note: PU-EdgeFormer geometry metrics were added after authenticity verification. "
        "PU-EdgeFormer PointNet++ classification is still pending. "
        "Delta figures include PU-EdgeFormer but do not include baseline as a method bar."
    )
    block_index.append("")

    if index.is_file():
        text = index.read_text(encoding="utf-8")
        marker = "## Geometry delta methods-only (with PU-EdgeFormer)"
        if marker in text:
            pre = text.split(marker)[0].rstrip()
            text = pre + "\n" + "\n".join(block_index)
        else:
            text = text.rstrip() + "\n" + "\n".join(block_index)
        index.write_text(text + "\n", encoding="utf-8")
    else:
        index.write_text("# Figure index\n" + "\n".join(block_index) + "\n", encoding="utf-8")

    cap_block = [
        "",
        "## Geometry delta methods-only with PU-EdgeFormer",
        "",
        CAPTION,
        "",
        "Methods on the x-axis: EAR, PDANS, PU-Net, PU-GCN, PU-EdgeFormer.",
        "PU-EdgeFormer geometry metrics were added after authenticity verification.",
        "PU-EdgeFormer PointNet++ classification is still pending.",
        "",
    ]
    if captions.is_file():
        ctext = captions.read_text(encoding="utf-8")
        marker = "## Geometry delta methods-only with PU-EdgeFormer"
        if marker in ctext:
            pre = ctext.split(marker)[0].rstrip()
            ctext = pre + "\n" + "\n".join(cap_block)
        else:
            ctext = ctext.rstrip() + "\n" + "\n".join(cap_block)
        captions.write_text(ctext + "\n", encoding="utf-8")
    else:
        captions.write_text("# Thesis figure captions\n" + "\n".join(cap_block) + "\n", encoding="utf-8")

    sum_block = [
        "",
        "## PU-EdgeFormer geometry update",
        "",
        "- Directory: `figures/modelnet40/geometry_delta_methods_only_with_pu_edgeformer/`",
        "- PU-EdgeFormer geometry metrics were added after authenticity verification.",
        "- PU-EdgeFormer PointNet++ classification is still pending.",
        "- Delta geometry figures include PU-EdgeFormer but do not include baseline as a method bar.",
        "- Absolute metrics use dense mesh surface / mesh / GT reference.",
        "",
    ]
    if summary.is_file():
        stext = summary.read_text(encoding="utf-8")
        marker = "## PU-EdgeFormer geometry update"
        if marker in stext:
            pre = stext.split(marker)[0].rstrip()
            stext = pre + "\n" + "\n".join(sum_block)
        else:
            stext = stext.rstrip() + "\n" + "\n".join(sum_block)
        summary.write_text(stext + "\n", encoding="utf-8")
    else:
        summary.write_text("# Visualization summary\n" + "\n".join(sum_block) + "\n", encoding="utf-8")


def main() -> int:
    deltas = load_deltas()
    for key, ylabel, stem in METRICS:
        plot_one(deltas, key, ylabel, stem)
        print(f"wrote {stem}")
    update_docs()
    print(f"figures in {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
