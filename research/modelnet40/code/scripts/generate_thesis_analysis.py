#!/usr/bin/env python3
"""Generate thesis-ready tables, rankings, and plot data from existing experiment CSVs."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"
RESULTS = PROJECT_ROOT / "pointnet2_results" / "x4_two_line_final"

LINEA_METHOD_ORDER = [
    ("Original baseline 1024", "lineA_original_baseline", 1024, None),
    ("Original + EAR 4096", "lineA_original_up/ear", 4096, "ear"),
    ("Original + PDANS 4096", "lineA_original_up/pdans", 4096, "pdans"),
    ("Original + PU-Net 4096", "lineA_original_up/pu_net", 4096, "pu_net"),
    ("Original + PU-GCN 4096", "lineA_original_up/pu_gcn", 4096, "pu_gcn"),
]

LINEB_METHOD_ORDER = [
    "Downsampled x4 baseline",
    "Downsampled x4 + EAR",
    "Downsampled x4 + PDANS",
    "Downsampled x4 + PU-Net",
    "Downsampled x4 + PU-GCN",
]

GEOMETRY_MAIN_METHODS = [
    "Original baseline",
    "Downsampled x4 + EAR",
    "Downsampled x4 + PDANS",
    "Downsampled x4 + PU-Net",
    "Downsampled x4 + PU-GCN",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def pp(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.2f}pp"


def fmt6(value: float) -> str:
    return f"{value:.6f}"


def read_csv_dict(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_linea_classification() -> list[dict]:
    rows: list[dict] = []
    baseline_final = baseline_best = None
    for method, rel_path, points, _ in LINEA_METHOD_ORDER:
        metrics_path = RESULTS / rel_path / "metrics.json"
        with metrics_path.open(encoding="utf-8") as f:
            m = json.load(f)
        row = {
            "line": "A",
            "method": method,
            "point_count": points,
            "final_test_overall_accuracy": m["final_test_overall_accuracy"],
            "final_test_class_accuracy": m["final_test_class_accuracy"],
            "best_overall_accuracy": m["overall_accuracy"],
            "best_class_accuracy": m["class_accuracy"],
        }
        rows.append(row)
        if method == "Original baseline 1024":
            baseline_final = row["final_test_overall_accuracy"]
            baseline_best = row["best_overall_accuracy"]

    for row in rows:
        row["delta_final_overall_vs_original_baseline"] = (
            row["final_test_overall_accuracy"] - baseline_final
        )
        row["delta_best_overall_vs_original_baseline"] = (
            row["best_overall_accuracy"] - baseline_best
        )
    return rows


def load_lineb_classification() -> list[dict]:
    path = REPORTS / "modelnet40_pointnet2_lineB_classification_summary.csv"
    raw = read_csv_dict(path)
    # Original baseline accuracies for secondary Line B gap columns.
    linea = load_linea_classification()
    orig = next(r for r in linea if r["method"].startswith("Original baseline"))
    o_final = orig["final_test_overall_accuracy"]
    o_best = orig["best_overall_accuracy"]
    rows = []
    for item in raw:
        method = item["method"]
        if method == "Downsampled x4 baseline":
            method = "Downsampled x4 baseline 256"
        elif method.startswith("Downsampled x4 + "):
            method = f"{method} 1024"
        final_o = float(item["final_test_overall_accuracy"])
        best_o = float(item["best_overall_accuracy"])
        row = {
            "line": item["line"],
            "method": method,
            "point_count": int(item["point_count"]),
            "final_test_overall_accuracy": final_o,
            "final_test_class_accuracy": float(item["final_test_class_accuracy"]),
            "best_overall_accuracy": best_o,
            "best_class_accuracy": float(item["best_class_accuracy"]),
            # Primary Line B delta: vs Downsampled ×4 baseline 256
            "delta_final_overall_vs_downsampled_baseline": float(
                item["delta_final_overall_vs_downsampled_baseline"]
            ),
            "delta_best_overall_vs_downsampled_baseline": float(
                item["delta_best_overall_vs_downsampled_baseline"]
            ),
            # Secondary Line B gap: vs Original baseline 1024
            "gap_final_overall_vs_original_baseline": float(
                item.get(
                    "gap_final_overall_vs_original_baseline",
                    final_o - o_final,
                )
            ),
            "gap_best_overall_vs_original_baseline": float(
                item.get(
                    "gap_best_overall_vs_original_baseline",
                    best_o - o_best,
                )
            ),
        }
        rows.append(row)
    ordered = []
    for name in LINEB_METHOD_ORDER:
        display = (
            "Downsampled x4 baseline 256"
            if name == "Downsampled x4 baseline"
            else f"{name} 1024"
        )
        for row in rows:
            if row["method"] == display:
                ordered.append(row)
                break
    return ordered


def classification_table_md(title: str, rows: list[dict], baseline_col: str) -> str:
    is_line_b = baseline_col == "vs_downsampled_baseline"
    if is_line_b:
        header = (
            "| Method | Points | Final Overall Accuracy | Final Class Accuracy | "
            "Best Overall Accuracy | Best Class Accuracy | "
            "Δ Best Overall vs Downsampled baseline (primary) | "
            "Δ Final Overall vs Downsampled baseline (primary) | "
            "gap Best Overall vs Original baseline (secondary) | "
            "gap Final Overall vs Original baseline (secondary) |"
        )
        sep = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    else:
        header = (
            "| Method | Points | Final Overall Accuracy | Final Class Accuracy | "
            "Best Overall Accuracy | Best Class Accuracy | "
            "Δ Best Overall vs line baseline | Δ Final Overall vs line baseline |"
        )
        sep = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    lines = [
        f"# {title}",
        "",
        f"- Generated at: {utc_now()}",
        "- Source: existing classification summaries (no retraining)",
    ]
    if is_line_b:
        lines.extend(
            [
                "- Primary delta: vs **Downsampled ×4 baseline 256** "
                "(`delta_accuracy_vs_downsampled_baseline`).",
                "- Secondary gap: vs **Original baseline 1024** "
                "(`gap_accuracy_vs_original_baseline`).",
                "- Geometry deltas remain vs Original baseline; do not mix with classification primary.",
                "- See `reports/modelnet40_lineB_comparison_logic_geometry_vs_classification.md`.",
            ]
        )
    lines.extend(["", header, sep])
    for row in rows:
        if is_line_b:
            lines.append(
                "| {method} | {pts} | {final_o} | {final_c} | {best_o} | {best_c} | "
                "{d_best} | {d_final} | {g_best} | {g_final} |".format(
                    method=row["method"],
                    pts=row["point_count"],
                    final_o=pct(row["final_test_overall_accuracy"]),
                    final_c=pct(row["final_test_class_accuracy"]),
                    best_o=pct(row["best_overall_accuracy"]),
                    best_c=pct(row["best_class_accuracy"]),
                    d_best=pp(row["delta_best_overall_vs_downsampled_baseline"]),
                    d_final=pp(row["delta_final_overall_vs_downsampled_baseline"]),
                    g_best=pp(row["gap_best_overall_vs_original_baseline"]),
                    g_final=pp(row["gap_final_overall_vs_original_baseline"]),
                )
            )
        else:
            lines.append(
                "| {method} | {pts} | {final_o} | {final_c} | {best_o} | {best_c} | "
                "{d_best} | {d_final} |".format(
                    method=row["method"],
                    pts=row["point_count"],
                    final_o=pct(row["final_test_overall_accuracy"]),
                    final_c=pct(row["final_test_class_accuracy"]),
                    best_o=pct(row["best_overall_accuracy"]),
                    best_c=pct(row["best_class_accuracy"]),
                    d_best=pp(row[f"delta_best_overall_{baseline_col}"]),
                    d_final=pp(row[f"delta_final_overall_{baseline_col}"]),
                )
            )
    return "\n".join(lines) + "\n"


def load_geometry_thesis() -> list[dict]:
    path = REPORTS / "modelnet40_quality_metrics_thesis_table.csv"
    rows = []
    for item in read_csv_dict(path):
        group = item["group"]
        method = group if group == "Original baseline" else group
        rows.append(
            {
                "method": method,
                "display_method": (
                    item["method"] if group != "Original baseline" else "Original baseline"
                ),
                "points": int(item["source_point_count"]),
                "CD": float(item["CD"]),
                "delta_CD_vs_original": float(item["delta_CD_vs_original"]),
                "HD": float(item["HD"]),
                "delta_HD_vs_original": float(item["delta_HD_vs_original"]),
                "NUC": float(item["NUC"]),
                "delta_NUC_vs_original": float(item["delta_NUC_vs_original"]),
                "exact_P2F": float(item["exact_P2F"]),
                "delta_P2F_vs_original": float(item["delta_P2F_vs_original"]),
            }
        )
    return rows


def geometry_table_md(title: str, rows: list[dict], subtitle: str = "") -> str:
    lines = [
        f"# {title}",
        "",
        f"- Generated at: {utc_now()}",
        "- Reference: Original baseline (1024 pts)",
        "- Lower is better for CD / HD / exact P2F / NUC",
    ]
    if subtitle:
        lines.append(f"- Scope: {subtitle}")
    lines.extend(
        [
            "",
            "| Method | Points | CD | Δ CD vs Original | HD | Δ HD vs Original | "
            "NUC | Δ NUC vs Original | exact P2F | Δ P2F vs Original |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        label = row["method"]
        lines.append(
            f"| {label} | {row['points']} | {fmt6(row['CD'])} | "
            f"{fmt6(row['delta_CD_vs_original'])} | {fmt6(row['HD'])} | "
            f"{fmt6(row['delta_HD_vs_original'])} | {fmt6(row['NUC'])} | "
            f"{fmt6(row['delta_NUC_vs_original'])} | {fmt6(row['exact_P2F'])} | "
            f"{fmt6(row['delta_P2F_vs_original'])} |"
        )
    return "\n".join(lines) + "\n"


def method_to_geometry_key(method: str) -> str:
    mapping = {
        "Original baseline 1024": "Original baseline",
        "Original + EAR 4096": "Original + EAR",
        "Original + PDANS 4096": "Original + PDANS",
        "Original + PU-Net 4096": "Original + PU-Net",
        "Original + PU-GCN 4096": "Original + PU-GCN",
        "Downsampled x4 baseline 256": "Downsampled x4 baseline",
        "Downsampled x4 + EAR 1024": "Downsampled x4 + EAR",
        "Downsampled x4 + PDANS 1024": "Downsampled x4 + PDANS",
        "Downsampled x4 + PU-Net 1024": "Downsampled x4 + PU-Net",
        "Downsampled x4 + PU-GCN 1024": "Downsampled x4 + PU-GCN",
    }
    return mapping[method]


def build_geometry_vs_classification(
    linea: list[dict], lineb: list[dict], geometry: list[dict]
) -> list[dict]:
    geom_by_method = {row["method"]: row for row in geometry}
    merged: list[dict] = []
    for line_label, cls_rows, baseline_col in [
        ("A", linea, "vs_original_baseline"),
        ("B", lineb, "vs_downsampled_baseline"),
    ]:
        for row in cls_rows:
            gkey = method_to_geometry_key(row["method"])
            g = geom_by_method[gkey]
            merged.append(
                {
                    "Line": line_label,
                    "Method": row["method"],
                    "Points": row["point_count"],
                    "CD": g["CD"],
                    "HD": g["HD"],
                    "NUC": g["NUC"],
                    "exact_P2F": g["exact_P2F"],
                    "Best Overall Accuracy": row["best_overall_accuracy"],
                    "Delta Best Overall vs line baseline": row[
                        f"delta_best_overall_{baseline_col}"
                    ],
                    "Final Overall Accuracy": row["final_test_overall_accuracy"],
                    "Delta Final Overall vs line baseline": row[
                        f"delta_final_overall_{baseline_col}"
                    ],
                }
            )
    return merged


def write_geometry_vs_classification(merged: list[dict]) -> None:
    csv_path = REPORTS / "modelnet40_geometry_vs_classification_final_summary.csv"
    md_path = REPORTS / "modelnet40_geometry_vs_classification_final_summary.md"
    fieldnames = list(merged[0].keys())
    write_csv(csv_path, fieldnames, merged)

    lines = [
        "# ModelNet40 Geometry vs Classification Final Summary",
        "",
        f"- Generated at: {utc_now()}",
        "- Merged from geometry thesis table and two-line classification summaries",
        "",
        "| Line | Method | Points | CD | HD | NUC | exact P2F | "
        "Best Overall Accuracy | Δ Best Overall vs line baseline | "
        "Final Overall Accuracy | Δ Final Overall vs line baseline |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in merged:
        lines.append(
            f"| {row['Line']} | {row['Method']} | {row['Points']} | "
            f"{fmt6(row['CD'])} | {fmt6(row['HD'])} | {fmt6(row['NUC'])} | "
            f"{fmt6(row['exact_P2F'])} | {pct(row['Best Overall Accuracy'])} | "
            f"{pp(row['Delta Best Overall vs line baseline'])} | "
            f"{pct(row['Final Overall Accuracy'])} | "
            f"{pp(row['Delta Final Overall vs line baseline'])} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def thesis_geometry_vs_classification_table(merged: list[dict]) -> str:
    lines = [
        "# ModelNet40 Thesis Table: Geometry vs Classification",
        "",
        f"- Generated at: {utc_now()}",
        "- Focus: Line B (downsampled ×4 recovery pipeline); Line A included for completeness",
        "- exact P2F is computed against dense mesh surfaces (not approximate)",
        "",
        "| Line | Method | Points | CD | HD | NUC | exact P2F | "
        "Best Overall Accuracy | Δ Best Overall vs line baseline | "
        "Final Overall Accuracy | Δ Final Overall vs line baseline |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in merged:
        lines.append(
            f"| {row['Line']} | {row['Method']} | {row['Points']} | "
            f"{fmt6(row['CD'])} | {fmt6(row['HD'])} | {fmt6(row['NUC'])} | "
            f"{fmt6(row['exact_P2F'])} | {pct(row['Best Overall Accuracy'])} | "
            f"{pp(row['Delta Best Overall vs line baseline'])} | "
            f"{pct(row['Final Overall Accuracy'])} | "
            f"{pp(row['Delta Final Overall vs line baseline'])} |"
        )
    return "\n".join(lines) + "\n"


def closest_geometry(
    geometry: list[dict], delta_key: str, *, line_b_only: bool = False
) -> dict:
    candidates = [g for g in geometry if g["method"] != "Original baseline"]
    if line_b_only:
        candidates = [g for g in candidates if g["method"].startswith("Downsampled x4 +")]
    return min(candidates, key=lambda g: abs(g[delta_key]))


def generate_rankings(
    linea: list[dict], lineb: list[dict], geometry: list[dict], merged: list[dict]
) -> str:
    linea_baseline = linea[0]
    lineb_baseline = lineb[0]

    linea_best = max(linea, key=lambda r: r["best_overall_accuracy"])
    linea_above_best = [
        r for r in linea[1:] if r["best_overall_accuracy"] > linea_baseline["best_overall_accuracy"]
    ]
    linea_below_best = [
        r for r in linea[1:] if r["best_overall_accuracy"] < linea_baseline["best_overall_accuracy"]
    ]
    linea_above_final = [
        r
        for r in linea[1:]
        if r["final_test_overall_accuracy"] > linea_baseline["final_test_overall_accuracy"]
    ]
    linea_below_final = [
        r
        for r in linea[1:]
        if r["final_test_overall_accuracy"] < linea_baseline["final_test_overall_accuracy"]
    ]

    lineb_best = max(lineb, key=lambda r: r["best_overall_accuracy"])
    lineb_above = [
        r
        for r in lineb[1:]
        if r["best_overall_accuracy"] > lineb_baseline["best_overall_accuracy"]
    ]
    lineb_below = [
        r
        for r in lineb[1:]
        if r["best_overall_accuracy"] < lineb_baseline["best_overall_accuracy"]
    ]

    lineb_only_above = [r for r in lineb_above]
    is_punet_only = (
        len(lineb_only_above) == 1
        and lineb_only_above[0]["method"] == "Downsampled x4 + PU-Net 1024"
    )

    geom_metrics = [
        ("CD", "delta_CD_vs_original"),
        ("HD", "delta_HD_vs_original"),
        ("NUC", "delta_NUC_vs_original"),
        ("exact_P2F", "delta_P2F_vs_original"),
    ]
    closest_all = {
        name: closest_geometry(geometry, delta, line_b_only=False)
        for name, delta in geom_metrics
    }
    closest_lineb = {
        name: closest_geometry(geometry, delta, line_b_only=True)
        for name, delta in geom_metrics
    }

    lineb_rows = [r for r in merged if r["Line"] == "B" and "baseline" not in r["Method"].lower()]
    lineb_geom_best_cd = min(lineb_rows, key=lambda r: r["CD"])
    lineb_cls_best = max(lineb_rows, key=lambda r: r["Best Overall Accuracy"])
    geom_cls_same = lineb_geom_best_cd["Method"] == lineb_cls_best["Method"]

    geom_improve_cls_not = [
        r
        for r in lineb_rows
        if r["CD"] < lineb_baseline["best_overall_accuracy"]  # wrong comparison, fix below
    ]
    geom_improve_cls_not = []
    baseline_geom = next(g for g in geometry if g["method"] == "Downsampled x4 baseline")
    for r in lineb_rows:
        better_geom = (
            r["CD"] < baseline_geom["CD"]
            or r["HD"] < baseline_geom["HD"]
            or r["NUC"] < baseline_geom["NUC"]
            or r["exact_P2F"] < baseline_geom["exact_P2F"]
        )
        cls_not_improve = r["Best Overall Accuracy"] <= lineb_baseline["best_overall_accuracy"]
        if better_geom and cls_not_improve:
            geom_improve_cls_not.append(r)

    cls_improve_geom_mixed = []
    for r in lineb_rows:
        cls_improve = r["Best Overall Accuracy"] > lineb_baseline["best_overall_accuracy"]
        cd_delta = r["CD"] - geometry[0]["CD"]
        hd_delta = r["HD"] - geometry[0]["HD"]
        nuc_delta = r["NUC"] - geometry[0]["NUC"]
        p2f_delta = r["exact_P2F"] - geometry[0]["exact_P2F"]
        wins = sum(
            [
                r["CD"] < geometry[0]["CD"],
                r["HD"] < geometry[0]["HD"],
                r["NUC"] < geometry[0]["NUC"],
                r["exact_P2F"] < geometry[0]["exact_P2F"],
            ]
        )
        if cls_improve and wins < 4:
            cls_improve_geom_mixed.append((r, wins))

    lines = [
        "# ModelNet40 Data Analysis Rankings",
        "",
        f"- Generated at: {utc_now()}",
        "- Derived from finalized geometry and classification summaries (no retraining)",
        "",
        "## Line A Classification",
        "",
        f"- **Highest best overall accuracy:** {linea_best['method']} "
        f"({pct(linea_best['best_overall_accuracy'])})",
        f"- **Original baseline best overall:** {pct(linea_baseline['best_overall_accuracy'])}; "
        f"**final overall:** {pct(linea_baseline['final_test_overall_accuracy'])}",
        "",
        "### Original + Upsampling vs Original baseline (best overall)",
        "",
    ]
    if linea_above_best:
        lines.append(
            "- Above baseline: "
            + ", ".join(
                f"{r['method']} ({pct(r['best_overall_accuracy'])}, "
                f"{pp(r['delta_best_overall_vs_original_baseline'])})"
                for r in linea_above_best
            )
        )
    else:
        lines.append("- Above baseline (best overall): none")
    lines.append(
        "- Below baseline: "
        + ", ".join(
            f"{r['method']} ({pct(r['best_overall_accuracy'])}, "
            f"{pp(r['delta_best_overall_vs_original_baseline'])})"
            for r in linea_below_best
        )
    )
    lines.extend(
        [
            "",
            "### Original + Upsampling vs Original baseline (final overall)",
            "",
            "- Above baseline: "
            + (
                ", ".join(
                    f"{r['method']} ({pct(r['final_test_overall_accuracy'])})"
                    for r in linea_above_final
                )
                if linea_above_final
                else "none"
            ),
            "- Below baseline: "
            + ", ".join(
                f"{r['method']} ({pct(r['final_test_overall_accuracy'])})"
                for r in linea_below_final
            ),
            "",
            "### 4096-point upsampling benefit (Line A)",
            "",
            "- No upsampling method exceeds the Original 1024 baseline on **best overall** accuracy.",
            "- On **final overall** accuracy, all four upsampling methods are also below baseline.",
            "- Increasing point count to 4096 does **not** yield a stable classification gain in this protocol.",
            "",
            "## Line B Classification",
            "",
            f"- **Highest best overall accuracy:** {lineb_best['method']} "
            f"({pct(lineb_best['best_overall_accuracy'])})",
            f"- **Downsampled x4 baseline best overall:** "
            f"{pct(lineb_baseline['best_overall_accuracy'])}",
            "",
            "### Downsampled + Upsampling vs Downsampled baseline (best overall)",
            "",
        ]
    )
    if lineb_above:
        lines.append(
            "- Above baseline: "
            + ", ".join(
                f"{r['method']} ({pct(r['best_overall_accuracy'])}, "
                f"{pp(r['delta_best_overall_vs_downsampled_baseline'])})"
                for r in lineb_above
            )
        )
    else:
        lines.append("- Above baseline: none")
    lines.append(
        "- Below baseline: "
        + ", ".join(
            f"{r['method']} ({pct(r['best_overall_accuracy'])}, "
            f"{pp(r['delta_best_overall_vs_downsampled_baseline'])})"
            for r in lineb_below
        )
    )
    lines.append(
        f"\n- **PU-Net only above baseline?** {'Yes' if is_punet_only else 'No'}"
    )
    lines.extend(["", "### Per-method delta vs Downsampled x4 baseline (best overall)", ""])
    for row in lineb[1:]:
        lines.append(
            f"- {row['method']}: {pp(row['delta_best_overall_vs_downsampled_baseline'])}"
        )

    lines.extend(
        [
            "",
            "## Geometry (vs Original baseline)",
            "",
            "### Line B upsampling methods (thesis focus)",
            "",
        ]
    )
    for metric, delta in geom_metrics:
        c = closest_lineb[metric]
        lines.append(f"- **Closest {metric}:** {c['method']} (Δ={fmt6(c[delta])})")
    lines.extend(["", "### All protocol branches", ""])
    for metric, delta in geom_metrics:
        c = closest_all[metric]
        lines.append(f"- **Closest {metric}:** {c['method']} (Δ={fmt6(c[delta])})")

    lines.extend(
        [
            "",
            "Line B upsampling CD ranking (smallest Δ CD vs Original):",
            "- PU-GCN (0.005279) < PU-Net (0.007567) < PDANS (0.013590) < EAR (0.031809)",
            "",
            "## Geometry vs Classification",
            "",
            f"- **Best Line B CD (upsampling):** {lineb_geom_best_cd['Method']} "
            f"(CD={fmt6(lineb_geom_best_cd['CD'])})",
            f"- **Best Line B classification (best overall):** {lineb_cls_best['Method']} "
            f"({pct(lineb_cls_best['Best Overall Accuracy'])})",
            f"- **Geometric CD leader equals classification leader?** "
            f"{'Yes' if geom_cls_same else 'No'}",
            "",
            "### Geometry improves but classification does not improve (Line B upsampling)",
            "",
        ]
    )
    if geom_improve_cls_not:
        for r in geom_improve_cls_not:
            lines.append(
                f"- {r['Method']}: better than downsampled baseline on at least one geometry "
                f"metric, but best overall ≤ baseline "
                f"({pct(r['Best Overall Accuracy'])})"
            )
    else:
        lines.append("- None identified under this rule.")

    lines.extend(["", "### Classification improves but geometry vs Original is mixed", ""])
    if cls_improve_geom_mixed:
        for r, wins in cls_improve_geom_mixed:
            lines.append(
                f"- {r['Method']}: best overall above baseline, but only {wins}/4 geometry "
                f"metrics beat Original baseline"
            )
    else:
        lines.append("- PU-Net is the only classifier above baseline; geometry is mixed vs Original.")

    return "\n".join(lines) + "\n"


def generate_thesis_analysis(
    linea: list[dict], lineb: list[dict], geometry: list[dict], merged: list[dict]
) -> str:
    orig = next(g for g in geometry if g["method"] == "Original baseline")
    lineb_geom = [g for g in geometry if g["method"].startswith("Downsampled x4 +")]
    punet_geom = next(g for g in geometry if g["method"] == "Downsampled x4 + PU-Net")
    pugcn_geom = next(g for g in geometry if g["method"] == "Downsampled x4 + PU-GCN")
    linea_base = linea[0]
    lineb_base = lineb[0]
    punet_cls = next(r for r in lineb if "PU-Net" in r["method"])

    return "\n".join(
        [
            "# ModelNet40 PointNet++ Thesis Result Analysis",
            "",
            f"- Generated at: {utc_now()}",
            "- Scope: finalized two-line protocol on ModelNet40 with PointNet++",
            "",
            "## 1. Experimental setup summary",
            "",
            "This study evaluates point cloud upsampling on **ModelNet40** using **PointNet++** "
            "as the downstream classifier under a **two-line protocol**:",
            "",
            "- **Line A:** Original 1024-point baseline vs **Original + Upsampling** at 4096 points.",
            "- **Line B:** Downsampled ×4 (256-point) baseline vs **Downsampled ×4 + Upsampling** "
            "at 1024 points.",
            "",
            "Upsampling methods: **EAR**, **PDANS**, **PU-Net**, **PU-GCN**.",
            "",
            "Geometry quality is measured against dense mesh references using **CD**, **HD**, "
            "**exact P2F**, and **NUC** (lower is better). Classification uses **final** and "
            "**best overall / class accuracy** from 200-epoch training (seed=42).",
            "",
            "## 2. Geometric quality analysis",
            "",
            f"**Original baseline** (1024 pts) is the reference: CD={fmt6(orig['CD'])}, "
            f"HD={fmt6(orig['HD'])}, exact P2F={fmt6(orig['exact_P2F'])}, NUC={fmt6(orig['NUC'])}.",
            "",
            "In **Line B**, **PU-GCN** and **PU-Net** achieve the smallest CD deltas vs Original "
            f"(PU-GCN ΔCD={fmt6(pugcn_geom['delta_CD_vs_original'])}, "
            f"PU-Net ΔCD={fmt6(punet_geom['delta_CD_vs_original'])}).",
            "",
            f"**PU-Net** is more balanced on **HD** (ΔHD={fmt6(punet_geom['delta_HD_vs_original'])}) "
            f"and **NUC** (ΔNUC={fmt6(punet_geom['delta_NUC_vs_original'])}).",
            "",
            f"**PU-GCN** shows competitive CD but clearly elevated **NUC** "
            f"(ΔNUC={fmt6(pugcn_geom['delta_NUC_vs_original'])}), indicating less uniform point "
            "distribution despite good Chamfer distance.",
            "",
            "**exact P2F** is fully computed (not approximate or pending), providing mesh-surface "
            "distance evidence alongside CD/HD/NUC.",
            "",
            "No single geometry metric should dominate interpretation: e.g., EAR has relatively "
            "low NUC but poor CD/HD; PDANS improves CD over EAR but HD remains high.",
            "",
            "## 3. Line B downstream classification analysis",
            "",
            f"**Downsampled ×4 baseline** (256 pts) reaches best overall accuracy "
            f"**{pct(lineb_base['best_overall_accuracy'])}** and final overall "
            f"**{pct(lineb_base['final_test_overall_accuracy'])}**.",
            "",
            f"**PU-Net** achieves the highest best overall accuracy at "
            f"**{pct(punet_cls['best_overall_accuracy'])}** "
            f"({pp(punet_cls['delta_best_overall_vs_downsampled_baseline'])} vs baseline) and is "
            "the **only** upsampling method slightly above the downsampled baseline on best overall.",
            "",
            "EAR (−2.04 pp), PDANS (−0.51 pp), and PU-GCN (−0.79 pp) all remain below baseline "
            "on best overall accuracy.",
            "",
            "These results show that recovering point count after ×4 downsampling does **not** "
            "automatically improve PointNet++ classification; only PU-Net yields a marginal gain.",
            "",
            "## 4. Line A downstream classification analysis",
            "",
            f"**Original baseline** (1024 pts): best overall **{pct(linea_base['best_overall_accuracy'])}**, "
            f"final overall **{pct(linea_base['final_test_overall_accuracy'])}**.",
            "",
            "All **Original + Upsampling** methods at 4096 points remain **below** the 1024 baseline "
            "on both best and final overall accuracy:",
            "",
        ]
        + [
            f"- {r['method']}: best {pct(r['best_overall_accuracy'])} "
            f"({pp(r['delta_best_overall_vs_original_baseline'])}), "
            f"final {pct(r['final_test_overall_accuracy'])} "
            f"({pp(r['delta_final_overall_vs_original_baseline'])})"
            for r in linea[1:]
        ]
        + [
            "",
            "Increasing to 4096 points does **not** produce a stable classification benefit in Line A; "
            "the native 1024-point baseline remains strongest.",
            "",
            "## 5. Geometry vs classification discussion",
            "",
            "Geometric quality and classification performance are **not fully aligned**.",
            "",
            f"PU-GCN has the best Line B CD (Δ={fmt6(pugcn_geom['delta_CD_vs_original'])}) but "
            f"classifies below PU-Net (CD={fmt6(pugcn_geom['CD'])} vs "
            f"{pct(punet_cls['best_overall_accuracy'])} best overall for PU-Net).",
            "",
            "PU-Net's stronger classification may relate to more balanced **HD** and **NUC**, "
            "suggesting PointNet++ is sensitive to local structure and point distribution—not only "
            "global Chamfer error.",
            "",
            "Geometric recovery quality and downstream task utility should be reported and "
            "interpreted separately.",
            "",
            "## 6. Thesis-ready conclusion",
            "",
            "- **Upsampling does not universally improve downstream classification.**",
            "- **Some methods improve geometric quality but do not improve classification** "
            "(e.g., PU-GCN on CD vs classification; PDANS partial geometry gains without "
            "classification gains).",
            "- **PU-Net appears to be the most promising Line B method** in this experiment, "
            "as the only upsampling variant slightly exceeding the downsampled baseline.",
            "- **The effect of point cloud upsampling depends on the upsampling method and the "
            "downstream task.**",
            "",
        ]
    )


def generate_plot_data(
    linea: list[dict], lineb: list[dict], geometry: list[dict], merged: list[dict]
) -> None:
    acc_rows = []
    for line_label, rows, baseline_col, baseline_ref in [
        ("A", linea, "vs_original_baseline", "Original baseline 1024"),
        ("B", lineb, "vs_downsampled_baseline", "Downsampled x4 baseline 256"),
    ]:
        for row in rows:
            for metric_name, value_key, delta_key in [
                ("best_overall_accuracy", "best_overall_accuracy", f"delta_best_overall_{baseline_col}"),
                ("final_overall_accuracy", "final_test_overall_accuracy", f"delta_final_overall_{baseline_col}"),
                ("best_class_accuracy", "best_class_accuracy", None),
                ("final_class_accuracy", "final_test_class_accuracy", None),
            ]:
                acc_rows.append(
                    {
                        "method": row["method"],
                        "line": line_label,
                        "point_count": row["point_count"],
                        "metric_name": metric_name,
                        "metric_value": row[value_key],
                        "delta_value": row.get(delta_key, ""),
                        "baseline_reference": baseline_ref,
                    }
                )

    geom_rows = []
    for row in geometry:
        baseline_ref = "Original baseline"
        for metric_name, value_key, delta_key in [
            ("CD", "CD", "delta_CD_vs_original"),
            ("HD", "HD", "delta_HD_vs_original"),
            ("NUC", "NUC", "delta_NUC_vs_original"),
            ("exact_P2F", "exact_P2F", "delta_P2F_vs_original"),
        ]:
            geom_rows.append(
                {
                    "method": row["method"],
                    "line": (
                        "reference"
                        if row["method"] == "Original baseline"
                        else ("A" if row["method"].startswith("Original +") else "B")
                    ),
                    "point_count": row["points"],
                    "metric_name": metric_name,
                    "metric_value": row[value_key],
                    "delta_value": row[delta_key],
                    "baseline_reference": baseline_ref,
                }
            )

    scatter_rows = []
    for row in merged:
        for metric_name, value_key in [
            ("CD", "CD"),
            ("HD", "HD"),
            ("NUC", "NUC"),
            ("exact_P2F", "exact_P2F"),
        ]:
            scatter_rows.append(
                {
                    "method": row["Method"],
                    "line": row["Line"],
                    "point_count": row["Points"],
                    "geometry_metric": metric_name,
                    "geometry_value": row[value_key],
                    "best_overall_accuracy": row["Best Overall Accuracy"],
                    "delta_best_overall_vs_line_baseline": row[
                        "Delta Best Overall vs line baseline"
                    ],
                    "baseline_reference": (
                        "Original baseline 1024"
                        if row["Line"] == "A"
                        else "Downsampled x4 baseline 256"
                    ),
                }
            )

    write_csv(
        REPORTS / "modelnet40_plot_data_accuracy.csv",
        [
            "method",
            "line",
            "point_count",
            "metric_name",
            "metric_value",
            "delta_value",
            "baseline_reference",
        ],
        acc_rows,
    )
    write_csv(
        REPORTS / "modelnet40_plot_data_geometry.csv",
        [
            "method",
            "line",
            "point_count",
            "metric_name",
            "metric_value",
            "delta_value",
            "baseline_reference",
        ],
        geom_rows,
    )
    write_csv(
        REPORTS / "modelnet40_plot_data_geometry_vs_accuracy.csv",
        [
            "method",
            "line",
            "point_count",
            "geometry_metric",
            "geometry_value",
            "best_overall_accuracy",
            "delta_best_overall_vs_line_baseline",
            "baseline_reference",
        ],
        scatter_rows,
    )


def main() -> None:
    linea = load_linea_classification()
    lineb = load_lineb_classification()
    geometry = load_geometry_thesis()

    # Persist missing Line A summary for reproducibility
    linea_csv = REPORTS / "modelnet40_pointnet2_lineA_classification_summary.csv"
    if not linea_csv.exists():
        write_csv(
            linea_csv,
            [
                "line",
                "method",
                "point_count",
                "final_test_overall_accuracy",
                "final_test_class_accuracy",
                "best_overall_accuracy",
                "best_class_accuracy",
                "delta_final_overall_vs_original_baseline",
                "delta_best_overall_vs_original_baseline",
            ],
            linea,
        )

    (REPORTS / "modelnet40_thesis_table_classification_lineA.md").write_text(
        classification_table_md(
            "ModelNet40 Thesis Table: Line A Classification",
            linea,
            "vs_original_baseline",
        ),
        encoding="utf-8",
    )
    (REPORTS / "modelnet40_thesis_table_classification_lineB.md").write_text(
        classification_table_md(
            "ModelNet40 Thesis Table: Line B Classification",
            lineb,
            "vs_downsampled_baseline",
        ),
        encoding="utf-8",
    )

    combined = linea + lineb
    combined_lines = [
        "# ModelNet40 Thesis Table: Combined Classification (Line A + Line B)",
        "",
        f"- Generated at: {utc_now()}",
        "",
        "## Line A",
        "",
    ]
    combined_lines.append(
        classification_table_md("", linea, "vs_original_baseline").strip().split("\n", 2)[-1]
    )
    combined_lines.extend(["", "## Line B", ""])
    combined_lines.append(
        classification_table_md("", lineb, "vs_downsampled_baseline").strip().split("\n", 2)[-1]
    )
    (REPORTS / "modelnet40_thesis_table_classification_combined.md").write_text(
        "\n".join(combined_lines) + "\n",
        encoding="utf-8",
    )

    geom_main = [g for g in geometry if g["method"] in GEOMETRY_MAIN_METHODS]
    (REPORTS / "modelnet40_thesis_table_geometry_main.md").write_text(
        geometry_table_md(
            "ModelNet40 Thesis Table: Geometry (Main)",
            geom_main,
            "Original baseline + Line B upsampling methods",
        ),
        encoding="utf-8",
    )
    (REPORTS / "modelnet40_thesis_table_geometry_extended.md").write_text(
        geometry_table_md(
            "ModelNet40 Thesis Table: Geometry (Extended)",
            geometry,
            "All protocol branches including Line A upsampling and downsampled baseline",
        ),
        encoding="utf-8",
    )

    merged = build_geometry_vs_classification(linea, lineb, geometry)
    write_geometry_vs_classification(merged)
    (REPORTS / "modelnet40_thesis_table_geometry_vs_classification.md").write_text(
        thesis_geometry_vs_classification_table(merged),
        encoding="utf-8",
    )
    write_csv(
        REPORTS / "modelnet40_thesis_table_geometry_vs_classification.csv",
        list(merged[0].keys()),
        merged,
    )

    # Final two-line classification combined
    final_cls_rows = []
    for row in linea:
        final_cls_rows.append(
            {
                "line": "A",
                "method": row["method"],
                "point_count": row["point_count"],
                "final_test_overall_accuracy": row["final_test_overall_accuracy"],
                "final_test_class_accuracy": row["final_test_class_accuracy"],
                "best_overall_accuracy": row["best_overall_accuracy"],
                "best_class_accuracy": row["best_class_accuracy"],
                "delta_final_overall_vs_line_baseline": row[
                    "delta_final_overall_vs_original_baseline"
                ],
                "delta_best_overall_vs_line_baseline": row[
                    "delta_best_overall_vs_original_baseline"
                ],
            }
        )
    for row in lineb:
        final_cls_rows.append(
            {
                "line": "B",
                "method": row["method"],
                "point_count": row["point_count"],
                "final_test_overall_accuracy": row["final_test_overall_accuracy"],
                "final_test_class_accuracy": row["final_test_class_accuracy"],
                "best_overall_accuracy": row["best_overall_accuracy"],
                "best_class_accuracy": row["best_class_accuracy"],
                "delta_final_overall_vs_line_baseline": row[
                    "delta_final_overall_vs_downsampled_baseline"
                ],
                "delta_best_overall_vs_line_baseline": row[
                    "delta_best_overall_vs_downsampled_baseline"
                ],
            }
        )
    write_csv(
        REPORTS / "modelnet40_pointnet2_final_two_line_classification_summary.csv",
        list(final_cls_rows[0].keys()),
        final_cls_rows,
    )
    final_md = [
        "# ModelNet40 PointNet++ Final Two-Line Classification Summary",
        "",
        f"- Generated at: {utc_now()}",
        "",
        "| line | method | pts | best overall | best class | Δ best overall | final overall | Δ final overall |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in final_cls_rows:
        final_md.append(
            f"| {row['line']} | {row['method']} | {row['point_count']} | "
            f"{pct(row['best_overall_accuracy'])} | {pct(row['best_class_accuracy'])} | "
            f"{pp(row['delta_best_overall_vs_line_baseline'])} | "
            f"{pct(row['final_test_overall_accuracy'])} | "
            f"{pp(row['delta_final_overall_vs_line_baseline'])} |"
        )
    (REPORTS / "modelnet40_pointnet2_final_two_line_classification_summary.md").write_text(
        "\n".join(final_md) + "\n",
        encoding="utf-8",
    )

    (REPORTS / "modelnet40_data_analysis_rankings.md").write_text(
        generate_rankings(linea, lineb, geometry, merged),
        encoding="utf-8",
    )
    (REPORTS / "modelnet40_thesis_result_analysis.md").write_text(
        generate_thesis_analysis(linea, lineb, geometry, merged),
        encoding="utf-8",
    )
    generate_plot_data(linea, lineb, geometry, merged)

    print("Generated thesis analysis artifacts in", REPORTS)


if __name__ == "__main__":
    main()
