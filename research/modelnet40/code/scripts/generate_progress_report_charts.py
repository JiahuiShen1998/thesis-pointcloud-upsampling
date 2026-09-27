#!/usr/bin/env python3
"""Figures for the findings that this report adds on top of the existing project figures."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "img")

NAVY, NAVY2, ACCENT, GREEN = "#10243E", "#1F4E79", "#B5443F", "#1E7A4C"
SLATE, RULE, ICE = "#5B7189", "#D3DDE8", "#EDF2F7"

plt.rcParams.update({
    "font.family": "Liberation Sans",
    "font.size": 10,
    "axes.edgecolor": SLATE, "axes.labelcolor": NAVY, "axes.titlecolor": NAVY,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
    "xtick.color": SLATE, "ytick.color": SLATE,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.bbox": "tight", "savefig.dpi": 200,
})


def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)
    ax.grid(axis="y", color=RULE, lw=0.7, zorder=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- data
GENUINE = [(256, 90.88, "Mesh-ref 256"), (1024, 91.95, "Original 1024"),
           (4096, 92.00, "Mesh-ref 4096")]
SYNTH = {
    256: [("Downsampled ×4", 90.85)],
    1024: [("PU-Net", 91.27), ("PDANS", 90.34), ("PU-GCN", 90.06),
           ("PU-EdgeFormer", 89.32), ("EAR", 88.81)],
    4096: [("PU-GCN", 91.63), ("PDANS", 91.62), ("EAR", 91.48),
           ("PU-Net", 90.95), ("PU-EdgeFormer", 90.54)],
}


# ---------------------------------------------------------------- 1. density curve
def density_curve():
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    despine(ax)
    xs = [g[0] for g in GENUINE]
    ys = [g[1] for g in GENUINE]
    ax.plot(xs, ys, "-o", color=NAVY, lw=2.4, ms=10, zorder=6,
            label="Genuine surface sampling")
    for (x, y, lab), dy in zip(GENUINE, (-20, 12, 12)):
        ax.annotate(f"{y:.2f} %", (x, y), textcoords="offset points", xytext=(0, dy),
                    ha="center", color=NAVY, fontweight="bold", fontsize=10.5, zorder=7)

    # synthetic points, with a small vertical dodge when two nearly coincide
    span = 92.55 - 88.3
    pt_per_unit = 3.05 * 72 / span          # axes height in points / data range
    for bucket, items in SYNTH.items():
        pts = sorted(items, key=lambda t: -t[1])
        prev_pt = None                       # previous label's y position, in points
        for name, oa in pts:
            ax.plot([bucket], [oa], "o", color=ACCENT, ms=6.5, alpha=.85, zorder=5)
            y_pt = oa * pt_per_unit
            if prev_pt is not None and prev_pt - y_pt < 11.0:
                y_pt = prev_pt - 11.0
            prev_pt = y_pt
            ax.annotate(name, (bucket, oa), textcoords="offset points",
                        xytext=(11, y_pt - oa * pt_per_unit - 3.2), ha="left",
                        color=ACCENT, fontsize=8.6, zorder=5)

    # gains, drawn under the curve so they never reach the title
    for (x0, y0, _), (x1, y1, _), col in ((GENUINE[0], GENUINE[1], GREEN),
                                          (GENUINE[1], GENUINE[2], ACCENT)):
        ax.text((x0 * x1) ** .5, min(y0, y1) - 0.42, f"+{y1 - y0:.2f} pp",
                ha="center", va="top", color=col, fontweight="bold", fontsize=11.5)

    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(["256", "1024", "4096"])
    ax.set_xlim(205, 9500)
    ax.set_ylim(88.3, 92.55)
    ax.set_xlabel("Points per cloud (a PointNet++ was trained from scratch at each count)")
    ax.set_ylabel("Best Overall Accuracy (%)")
    ax.set_title("Real density pays once, then stops —\nand no synthetic cloud reaches the curve",
                 pad=14)
    ax.legend(handles=[Line2D([], [], color=NAVY, marker="o", lw=2.4, ms=8,
                              label="Genuine surface sampling"),
                       Line2D([], [], color=ACCENT, marker="o", lw=0, ms=7,
                              label="Upsampled / decimated")],
              frameon=False, loc="lower right", fontsize=9.5)
    fig.savefig(os.path.join(OUT, "chart_density_curve.png"))
    plt.close(fig)


# ---------------------------------------------------------------- 2. SA1 reception
def sa1_reception():
    d = json.load(open(os.path.join(HERE, "sa1_probe.json")))["results"]
    keys = [("256|Mesh-ref 256", "256", "90.88 %"),
            ("1024|Original", "1024", "91.95 %"),
            ("4096|Mesh-ref 4096", "4096", "92.00 %")]
    pad = [d[k][2] * 100 for k, _, _ in keys]
    disc = [d[k][1] * 100 for k, _, _ in keys]
    x = list(range(len(keys)))
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    despine(ax)
    w = 0.33
    ax.bar([i - w / 2 - 0.015 for i in x], pad, w, color=NAVY2, zorder=3,
           label="Neighbour slots filled by a repeated point  (information missing)")
    ax.bar([i + w / 2 + 0.015 for i in x], disc, w, color=ACCENT, zorder=3,
           label="In-ball points discarded by the 32-cap  (information unreadable)")
    for i, (pv, dv) in enumerate(zip(pad, disc)):
        ax.text(i - w / 2 - 0.015, pv + 2.0, f"{pv:.1f} %", ha="center",
                color=NAVY2, fontweight="bold", fontsize=10)
        ax.text(i + w / 2 + 0.015, dv + 2.0, f"{dv:.1f} %", ha="center",
                color=ACCENT, fontweight="bold", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab} points\nBest OA {oa}" for _, lab, oa in keys],
                       fontweight="bold")
    ax.set_ylabel("Share of SA1 intake (%)")
    ax.set_ylim(0, 96)
    ax.set_title("The two ways a cloud fails PointNet++ SA1 trade places with density\n"
                 "(npoint 512, radius 0.2, nsample 32 — genuine clouds only)", pad=12)
    ax.legend(frameon=False, fontsize=9.2, loc="upper center", bbox_to_anchor=(0.5, 0.99))
    fig.savefig(os.path.join(OUT, "chart_sa1_reception.png"))
    plt.close(fig)


# ---------------------------------------------------------------- 3. rank flip slopegraph
def rank_flip():
    old = {"PDANS": 0.0259, "EAR": 0.0292, "PU-EdgeFormer": 0.0303,
           "PU-GCN": 0.0347, "PU-Net": 0.0520}
    new = {"PU-GCN": 0.0423, "PU-EdgeFormer": 0.0467, "PDANS": 0.0468,
           "PU-Net": 0.0497, "EAR": 0.0547}
    ro = {m: i + 1 for i, m in enumerate(sorted(old, key=old.get))}
    rn = {m: i + 1 for i, m in enumerate(sorted(new, key=new.get))}
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for m in ro:
        moved = ro[m] != rn[m]
        col = ACCENT if abs(ro[m] - rn[m]) >= 2 else (SLATE if moved else NAVY)
        ax.plot([0, 1], [ro[m], rn[m]], "-o", color=col, lw=2.2, ms=7,
                alpha=.95, zorder=4)
        ax.text(-0.045, ro[m], f"{m}  {old[m]:.4f}", ha="right", va="center",
                color=col, fontsize=9.5, fontweight="bold")
        ax.text(1.045, rn[m], f"{new[m]:.4f}  {m}", ha="left", va="center",
                color=col, fontsize=9.5, fontweight="bold")
    ax.set_xlim(-0.72, 1.72)
    ax.set_ylim(5.6, 0.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["CD vs Original 1024\n(reference = the input)",
                        "CD vs mesh-ref 4096\n(equal-N)"], fontweight="bold")
    ax.set_yticks(range(1, 6))
    ax.set_ylabel("Rank by Chamfer Distance")
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Line A: the ranking is an artefact of the reference set")
    fig.savefig(os.path.join(OUT, "chart_rank_flip.png"))
    plt.close(fig)


# ---------------------------------------------------------------- 4. CD decomposition
def decomposition():
    d = json.load(open(os.path.join(HERE, "cardinality_probe.json")))["results"]
    names = {"repeat_x4": "Repeat ×4\n(does nothing)", "pdans": "PDANS", "ear": "EAR",
             "pu_edgeformer": "PU-EdgeFormer", "pu_gcn": "PU-GCN", "pu_net": "PU-Net"}
    order = sorted(names, key=lambda k: d[k][2])
    fwd = [d[k][0] for k in order]
    bwd = [d[k][1] for k in order]
    x = range(len(order))
    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    ax.set_facecolor("#FBF4F3")               # tinted ground = a rejected measurement
    despine(ax)
    ax.bar(x, fwd, 0.62, color=ACCENT, zorder=3,
           label="forward — output point → nearest Original  (grows when gaps are filled)")
    ax.bar(x, bwd, 0.62, bottom=fwd, color=NAVY2, zorder=3,
           label="backward — Original point → nearest output  (zero if the input is kept)")
    for i, k in enumerate(order):
        tot = d[k][2]
        ax.text(i, tot + 0.0016, "0.0000" if tot < 1e-9 else f"{tot:.4f}", ha="center",
                color=ACCENT if tot < 1e-9 else NAVY, fontweight="bold", fontsize=9.5)
    ax.text(0, 0.0092, "best score\nobtainable", ha="center", va="bottom", color=ACCENT,
            fontsize=8.6, style="italic", zorder=5)
    ax.set_xticks(list(x))
    ax.set_xticklabels([names[k] for k in order], fontsize=9.5)
    ax.set_ylabel("CD:  4096-point output  vs  1024-point reference")
    ax.set_ylim(0, 0.060)
    ax.set_title("SUPERSEDED MEASUREMENT — shown here only to be refuted\n"
                 "Every bar is 4096 points scored against a 1024-point reference",
                 color=ACCENT, fontsize=11.5, pad=10)
    ax.legend(frameon=False, fontsize=8.8, loc="upper left")
    fig.savefig(os.path.join(OUT, "chart_cd_decomposition.png"))
    plt.close(fig)


# ---------------------------------------------------------------- 5. geometry vs accuracy per bucket
def geom_vs_acc():
    b1024 = [("PU-Net", .0646, .1083, .003, 91.27), ("PDANS", .0627, .1905, .031, 90.34),
             ("PU-GCN", .0558, .1514, .440, 90.06), ("PU-EdgeFormer", .0619, .1776, .039, 89.32),
             ("EAR", .0767, .1932, .118, 88.81)]
    b4096 = [("PU-GCN", .0423, .0932, .121, 91.63), ("PDANS", .0468, .1135, .019, 91.62),
             ("EAR", .0547, .1154, .174, 91.48), ("PU-Net", .0497, .1112, .037, 90.95),
             ("PU-EdgeFormer", .0467, .1109, .034, 90.54)]
    rho = {("1024", "CD"): .10, ("1024", "HD"): .70, ("1024", "|ΔNUC|"): .70,
           ("4096", "CD"): .30, ("4096", "HD"): .10, ("4096", "|ΔNUC|"): -.20}
    fig, axes = plt.subplots(2, 3, figsize=(9.4, 5.4))
    for r, (bucket, data) in enumerate((("1024", b1024), ("4096", b4096))):
        for c, (lab, idx) in enumerate((("CD", 1), ("HD", 2), ("|ΔNUC|", 3))):
            ax = axes[r][c]
            despine(ax)
            xs = [d[idx] for d in data]
            ys = [d[4] for d in data]
            strong = abs(rho[(bucket, lab)]) >= 0.7
            ax.scatter(xs, ys, s=62, color=NAVY if strong else SLATE,
                       zorder=4, alpha=.9 if strong else .55)
            for d in data:
                ax.annotate(d[0].replace("PU-EdgeFormer", "PU-EF"), (d[idx], d[4]),
                            textcoords="offset points", xytext=(6, -3),
                            fontsize=7.4, color=NAVY if strong else SLATE)
            ax.set_title(f"{bucket} pts · {lab}   ρ = {rho[(bucket, lab)]:+.2f}",
                         fontsize=10, color=NAVY if strong else SLATE,
                         fontweight="bold" if strong else "normal")
            ax.tick_params(labelsize=8)
            if c == 0:
                ax.set_ylabel("Best OA (%)", fontsize=9)
            ax.margins(x=.28, y=.30)
    fig.suptitle("Spearman rank correlation with accuracy, per point-count bucket "
                 "(n = 5 methods; descriptive only)",
                 fontsize=11.5, fontweight="bold", color=NAVY, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "chart_geom_vs_acc.png"))
    plt.close(fig)


if __name__ == "__main__":
    density_curve(); sa1_reception(); rank_flip(); decomposition(); geom_vs_acc()
    for f in sorted(os.listdir(OUT)):
        if f.startswith("chart_"):
            print(" ", f, round(os.path.getsize(os.path.join(OUT, f)) / 1024), "KB")
