#!/usr/bin/env python3
"""Generate thesis/presentation interactive 3D point cloud HTML viewers (v2).

Reads existing .npy point clouds only — no training, no metric recomputation, no data modification.
"""

from __future__ import annotations

import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = PROJECT_ROOT / "datasets"
OUT_ROOT = PROJECT_ROOT / "figures" / "modelnet40" / "pointcloud_examples_interactive_v2"
REPORTS = PROJECT_ROOT / "reports"

PREFERRED_CLASSES = ["airplane", "chair", "table", "car", "sofa"]
PREFERRED_SAMPLES = {
    "airplane": "airplane_0627",
    "chair": "chair_0890",
    "table": "table_0393",
    "car": "car_0198",
    "sofa": "sofa_0681",
}
SINGLE_METHOD_CLASSES = ["airplane", "chair"]

MARKER_SIZE = {256: 2.0, 1024: 1.2, 4096: 0.8}
DEFAULT_CAMERA = dict(eye=dict(x=1.6, y=1.6, z=1.1), up=dict(x=0, y=0, z=1))
AXIS_STYLE = dict(
    showbackground=False,
    showgrid=False,
    showline=False,
    showticklabels=False,
    zeroline=False,
    title="",
)

GENERATED: list[str] = []
FAILURES: list[str] = []
SELECTED_SAMPLES: list[dict] = []
HAS_KALEIDO = False

try:
    import kaleido  # noqa: F401

    HAS_KALEIDO = True
except ImportError:
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def marker_size(n: int) -> float:
    if n in MARKER_SIZE:
        return MARKER_SIZE[n]
    if n < 512:
        return MARKER_SIZE[256]
    if n < 2048:
        return MARKER_SIZE[1024]
    return MARKER_SIZE[4096]


def load_points(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    pts = np.load(path)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return None
    return pts.astype(np.float32)


def compute_axis_limits(point_sets: list[np.ndarray]) -> tuple[list[float], list[float], list[float]]:
    all_pts = np.vstack(point_sets)
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0 * 1.05)
    if radius <= 0:
        radius = 0.5
    return (
        [center[0] - radius, center[0] + radius],
        [center[1] - radius, center[1] + radius],
        [center[2] - radius, center[2] + radius],
    )


def scene_layout(xrange: list[float], yrange: list[float], zrange: list[float]) -> dict:
    return dict(
        xaxis={**AXIS_STYLE, "range": xrange},
        yaxis={**AXIS_STYLE, "range": yrange},
        zaxis={**AXIS_STYLE, "range": zrange},
        aspectmode="cube",
        camera=DEFAULT_CAMERA,
    )


def make_scatter_trace(
    pts: np.ndarray,
    n_points: int,
    color_mode: str,
    name: str,
    visible: bool = True,
) -> go.Scatter3d:
    size = marker_size(n_points)
    if color_mode == "black":
        marker = dict(size=size, color="#111111", opacity=0.92)
    else:
        marker = dict(
            size=size,
            color=pts[:, 2],
            colorscale="Viridis",
            opacity=0.92,
            showscale=False,
        )
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode="markers",
        marker=marker,
        name=name,
        visible=visible,
        hovertemplate=f"{name}<br>n={n_points}<extra></extra>",
    )


def page_info(
    line: str,
    class_name: str,
    sample_id: str,
    panels: list[tuple[str, np.ndarray, int]],
) -> str:
    counts = ", ".join(f"{title}: {n}" for title, _, n in panels)
    return (
        f"Dataset: ModelNet40 | Class: {class_name} | Sample: {sample_id} | "
        f"Protocol: Line {line} | Point counts: {counts} | "
        f"Interaction: rotate / zoom / pan (drag to rotate, scroll to zoom)"
    )


def write_html(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(path),
        include_plotlyjs="cdn",
        config=dict(displayModeBar=True, scrollZoom=True),
    )
    GENERATED.append(str(path))


def try_preview_png(fig: go.Figure, path: Path) -> bool:
    if not HAS_KALEIDO:
        return False
    try:
        png_path = path.with_suffix(".png")
        fig.write_image(str(png_path), width=1200, height=700, scale=2)
        GENERATED.append(str(png_path))
        return True
    except Exception as exc:
        FAILURES.append(f"Preview PNG failed for {path.name}: {exc}")
        return False


def resolve_samples() -> list[tuple[str, str]]:
    split = "test"
    original_root = DATASETS / "modelnet40_original" / split
    selected: list[tuple[str, str]] = []
    fallbacks: list[str] = []

    for class_name in PREFERRED_CLASSES:
        class_dir = original_root / class_name
        if not class_dir.is_dir():
            alt = sorted(
                d.name
                for d in original_root.iterdir()
                if d.is_dir() and (d / f"{d.name}_0001.npy").exists()
            )
            if alt:
                class_name = alt[0]
                fallbacks.append(f"Preferred class missing; using {class_name}")
            else:
                FAILURES.append(f"No fallback class found when {class_name} missing")
                continue

        sample_id = PREFERRED_SAMPLES.get(class_name)
        if sample_id is None or not (class_dir / f"{sample_id}.npy").exists():
            files = sorted(class_dir.glob("*.npy"))
            if not files:
                FAILURES.append(f"No .npy files in {class_dir}")
                continue
            sample_id = files[0].stem
            fallbacks.append(f"{class_name}: using fallback sample {sample_id}")

        selected.append((class_name, sample_id))
        SELECTED_SAMPLES.append(
            {
                "class": class_name,
                "sample_id": sample_id,
                "fallback_notes": fallbacks[-1] if fallbacks and fallbacks[-1].startswith(class_name) else "",
            }
        )

    return selected


def lineb_panel_specs(class_name: str, sample_id: str) -> list[tuple[str, Path, int, str]]:
    split = "test"
    fname = f"{sample_id}.npy"
    return [
        ("Original 1024", DATASETS / "modelnet40_original" / split / class_name / fname, 1024, "original"),
        (
            "Downsampled x4 256",
            DATASETS / "modelnet40_downsampled_x4" / split / class_name / fname,
            256,
            "downsampled",
        ),
        (
            "EAR 1024",
            DATASETS / "lineB_downsampled_x4_up" / "strict_N" / "ear" / split / class_name / fname,
            1024,
            "ear",
        ),
        (
            "PDANS 1024",
            DATASETS / "lineB_downsampled_x4_up" / "strict_N" / "pdans" / split / class_name / fname,
            1024,
            "pdans",
        ),
        (
            "PU-Net 1024",
            DATASETS / "lineB_downsampled_x4_up" / "strict_N" / "pu_net" / split / class_name / fname,
            1024,
            "pu_net",
        ),
        (
            "PU-GCN 1024",
            DATASETS / "lineB_downsampled_x4_up" / "strict_N" / "pu_gcn" / split / class_name / fname,
            1024,
            "pu_gcn",
        ),
    ]


def linea_panel_specs(class_name: str, sample_id: str) -> list[tuple[str, Path, int, str]]:
    split = "test"
    fname = f"{sample_id}.npy"
    return [
        ("Original 1024", DATASETS / "modelnet40_original" / split / class_name / fname, 1024, "original"),
        (
            "EAR 4096",
            DATASETS / "lineA_original_up" / "strict_4N" / "ear" / split / class_name / fname,
            4096,
            "ear",
        ),
        (
            "PDANS 4096",
            DATASETS / "lineA_original_up" / "strict_4N" / "pdans" / split / class_name / fname,
            4096,
            "pdans",
        ),
        (
            "PU-Net 4096",
            DATASETS / "lineA_original_up" / "strict_4N" / "pu_net" / split / class_name / fname,
            4096,
            "pu_net",
        ),
        (
            "PU-GCN 4096",
            DATASETS / "lineA_original_up" / "strict_4N" / "pu_gcn" / split / class_name / fname,
            4096,
            "pu_gcn",
        ),
    ]


def load_panels(specs: list[tuple[str, Path, int, str]]) -> list[tuple[str, np.ndarray, int, str]]:
    panels = []
    for title, path, n, slug in specs:
        pts = load_points(path)
        if pts is None:
            raise FileNotFoundError(path)
        panels.append((title, pts, n, slug))
    return panels


def build_grid_html(
    panels: list[tuple[str, np.ndarray, int, str]],
    line: str,
    class_name: str,
    sample_id: str,
    color_mode: str,
    out_path: Path,
) -> None:
    cols = len(panels)
    subplot_titles = [f"{t} ({n} pts)" for t, _, n, _ in panels]
    fig = make_subplots(
        rows=1,
        cols=cols,
        specs=[[{"type": "scatter3d"}] * cols],
        subplot_titles=subplot_titles,
        horizontal_spacing=0.02,
    )

    point_sets = [p for _, p, _, _ in panels]
    xrange, yrange, zrange = compute_axis_limits(point_sets)

    for i, (title, pts, n, _) in enumerate(panels, start=1):
        fig.add_trace(
            make_scatter_trace(pts, n, color_mode, title),
            row=1,
            col=i,
        )

    layout_updates = dict(
        title=dict(
            text=(
                f"<b>Line {line} Point Cloud Comparison — {class_name} / {sample_id}</b>"
                f"<br><sup>{page_info(line, class_name, sample_id, [(t, p, n) for t, p, n, _ in panels])}</sup>"
            ),
            x=0.5,
            xanchor="center",
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        margin=dict(l=10, r=10, t=100, b=20),
        height=480,
        width=max(320 * cols, 900),
    )
    fig.update_layout(**layout_updates)

    for i in range(1, cols + 1):
        scene_key = "scene" if i == 1 else f"scene{i}"
        fig.update_layout(**{scene_key: scene_layout(xrange, yrange, zrange)})

    write_html(fig, out_path)
    try_preview_png(fig, out_path)


def build_dropdown_html(
    panels: list[tuple[str, np.ndarray, int, str]],
    line: str,
    class_name: str,
    sample_id: str,
    out_path: Path,
    color_mode: str = "depth",
) -> None:
    point_sets = [p for _, p, _, _ in panels]
    xrange, yrange, zrange = compute_axis_limits(point_sets)

    fig = go.Figure()
    for i, (title, pts, n, _) in enumerate(panels):
        fig.add_trace(make_scatter_trace(pts, n, color_mode, title, visible=(i == 0)))

    buttons = []
    for i, (title, _, n, _) in enumerate(panels):
        visibility = [j == i for j in range(len(panels))]
        buttons.append(
            dict(
                label=f"{title} ({n} pts)",
                method="update",
                args=[
                    {"visible": visibility},
                    {"title": f"<b>{title}</b> — {class_name} / {sample_id} (Line {line})"},
                ],
            )
        )

    fig.update_layout(
        title=dict(
            text=(
                f"<b>Line {line} Point Cloud — {class_name} / {sample_id}</b>"
                f"<br><sup>{page_info(line, class_name, sample_id, [(t, p, n) for t, p, n, _ in panels])}</sup>"
            ),
            x=0.5,
            xanchor="center",
        ),
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.12,
                yanchor="top",
                bgcolor="white",
                bordercolor="#cccccc",
            )
        ],
        scene=scene_layout(xrange, yrange, zrange),
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=120, b=0),
        height=640,
        width=900,
    )
    write_html(fig, out_path)
    try_preview_png(fig, out_path)


def build_single_method_html(
    title: str,
    pts: np.ndarray,
    n: int,
    line: str,
    class_name: str,
    sample_id: str,
    slug: str,
    out_path: Path,
    color_mode: str = "depth",
) -> None:
    fig = go.Figure(data=[make_scatter_trace(pts, n, color_mode, title)])
    xrange, yrange, zrange = compute_axis_limits([pts])
    fig.update_layout(
        title=dict(
            text=(
                f"<b>{title} — {class_name} / {sample_id} (Line {line})</b>"
                f"<br><sup>Dataset: ModelNet40 | {n} points | Interaction: rotate / zoom / pan</sup>"
            ),
            x=0.5,
            xanchor="center",
        ),
        scene=scene_layout(xrange, yrange, zrange),
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=90, b=0),
        height=720,
        width=960,
    )
    write_html(fig, out_path)
    try_preview_png(fig, out_path)


def process_sample_lineb(class_name: str, sample_id: str) -> None:
    specs = lineb_panel_specs(class_name, sample_id)
    try:
        panels = load_panels(specs)
    except FileNotFoundError as exc:
        FAILURES.append(f"Line B {class_name}/{sample_id}: missing {exc}")
        return

    stem = f"lineB_{class_name}_{sample_id}"
    grid_dir = OUT_ROOT / "lineB"
    for mode in ("black", "depth"):
        build_grid_html(
            panels,
            "B",
            class_name,
            sample_id,
            mode,
            grid_dir / f"{stem}_interactive_grid_{mode}.html",
        )

    build_dropdown_html(
        panels,
        "B",
        class_name,
        sample_id,
        OUT_ROOT / "dropdown" / f"{stem}_dropdown.html",
    )

    if class_name in SINGLE_METHOD_CLASSES:
        for title, pts, n, slug in panels:
            build_single_method_html(
                title,
                pts,
                n,
                "B",
                class_name,
                sample_id,
                slug,
                OUT_ROOT / "single_method" / f"{stem}_{slug}.html",
            )


def process_sample_linea(class_name: str, sample_id: str) -> None:
    specs = linea_panel_specs(class_name, sample_id)
    try:
        panels = load_panels(specs)
    except FileNotFoundError as exc:
        FAILURES.append(f"Line A {class_name}/{sample_id}: missing {exc}")
        return

    stem = f"lineA_{class_name}_{sample_id}"
    grid_dir = OUT_ROOT / "lineA"
    for mode in ("black", "depth"):
        build_grid_html(
            panels,
            "A",
            class_name,
            sample_id,
            mode,
            grid_dir / f"{stem}_interactive_grid_{mode}.html",
        )

    build_dropdown_html(
        panels,
        "A",
        class_name,
        sample_id,
        OUT_ROOT / "dropdown" / f"{stem}_dropdown.html",
    )

    if class_name in SINGLE_METHOD_CLASSES:
        for title, pts, n, slug in panels:
            build_single_method_html(
                title,
                pts,
                n,
                "A",
                class_name,
                sample_id,
                slug,
                OUT_ROOT / "single_method" / f"{stem}_{slug}.html",
            )


def write_audit_report() -> None:
    linea_grids = sorted((OUT_ROOT / "lineA").glob("*.html"))
    lineb_grids = sorted((OUT_ROOT / "lineB").glob("*.html"))
    dropdowns = sorted((OUT_ROOT / "dropdown").glob("*.html"))
    singles = sorted((OUT_ROOT / "single_method").glob("*.html"))

    black_count = sum(1 for p in GENERATED if p.endswith("_black.html"))
    depth_count = sum(1 for p in GENERATED if p.endswith("_depth.html"))

    lines = [
        "# ModelNet40 Interactive Point Cloud v2 Audit",
        "",
        f"- Generated at: {utc_now()}",
        f"- Project root: `{PROJECT_ROOT}`",
        f"- Output root: `{OUT_ROOT}`",
        "",
        "## Selected samples",
        "",
        "| Class | Sample ID | Notes |",
        "| --- | --- | --- |",
    ]
    for s in SELECTED_SAMPLES:
        lines.append(f"| {s['class']} | {s['sample_id']} | {s.get('fallback_notes', '') or 'preferred sample'} |")

    lines.extend(
        [
            "",
            "## Coverage",
            "",
            f"- **Line A grid HTML (black + depth):** {len(linea_grids)} files "
            f"(expected {len(SELECTED_SAMPLES) * 2})",
            f"- **Line B grid HTML (black + depth):** {len(lineb_grids)} files "
            f"(expected {len(SELECTED_SAMPLES) * 2})",
            f"- **Dropdown HTML:** {len(dropdowns)} files (expected {len(SELECTED_SAMPLES) * 2})",
            f"- **Single-method HTML:** {len(singles)} files "
            f"(expected {len(SINGLE_METHOD_CLASSES) * (5 + 6)} = 22 for airplane + chair)",
            "",
            "## Color versions",
            "",
            f"- Black grid versions generated: {black_count >= len(SELECTED_SAMPLES) * 2}",
            f"- Depth grid versions generated: {depth_count >= len(SELECTED_SAMPLES) * 2}",
            "",
            "## Plotting configuration",
            "",
            "- **Library:** Plotly 3D scatter (`Scatter3d`)",
            f"- **Marker sizes:** 256 pts → {MARKER_SIZE[256]}, 1024 pts → {MARKER_SIZE[1024]}, "
            f"4096 pts → {MARKER_SIZE[4096]}",
            "- **Shared axis limits:** computed from all methods for the same sample (fair comparison)",
            "- **Shared camera:** identical initial eye/up for all subplots",
            "- **Data modified:** **NO** — raw `.npy` coordinates loaded without interpolation or duplication",
            f"- **Static preview PNG:** {'attempted via kaleido' if HAS_KALEIDO else 'skipped (kaleido not installed)'}",
            "",
            "## Generated HTML paths",
            "",
            "### Line A grid",
            "",
        ]
    )
    lines.extend(f"- `{p.relative_to(PROJECT_ROOT)}`" for p in linea_grids)
    lines.extend(["", "### Line B grid", ""])
    lines.extend(f"- `{p.relative_to(PROJECT_ROOT)}`" for p in lineb_grids)
    lines.extend(["", "### Dropdown", ""])
    lines.extend(f"- `{p.relative_to(PROJECT_ROOT)}`" for p in dropdowns)
    lines.extend(["", "### Single-method", ""])
    lines.extend(f"- `{p.relative_to(PROJECT_ROOT)}`" for p in singles)

    lines.extend(["", "## Failures", ""])
    if FAILURES:
        lines.extend(f"- {f}" for f in FAILURES)
    else:
        lines.append("- None")

    (REPORTS / "modelnet40_interactive_pointcloud_v2_audit.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def update_figure_index() -> None:
    index_path = PROJECT_ROOT / "figures" / "modelnet40" / "figure_index.md"
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else ""
    if "pointcloud_examples_interactive_v2" in existing:
        # Replace section if re-run
        start = existing.find("## Interactive point cloud viewers (v2)")
        if start >= 0:
            existing = existing[:start].rstrip() + "\n"

    block = textwrap.dedent(
        f"""
        ## Interactive point cloud viewers (v2)

        - Generated at: {utc_now()}
        - Root: `{OUT_ROOT}`
        - Purpose: qualitative inspection and presentation (rotate / zoom / pan)
        - Static PNG/PDF figures in `pointcloud_examples/` remain for thesis document insertion

        | HTML pattern | Description | Thesis section |
        | --- | --- | --- |
        | `pointcloud_examples_interactive_v2/lineA/lineA_<class>_<id>_interactive_grid_black.html` | Line A 5-panel grid, single-color markers | Qualitative — Line A |
        | `pointcloud_examples_interactive_v2/lineA/lineA_<class>_<id>_interactive_grid_depth.html` | Line A 5-panel grid, depth-colored markers | Qualitative — Line A |
        | `pointcloud_examples_interactive_v2/lineB/lineB_<class>_<id>_interactive_grid_black.html` | Line B 6-panel grid, single-color markers | Qualitative — Line B |
        | `pointcloud_examples_interactive_v2/lineB/lineB_<class>_<id>_interactive_grid_depth.html` | Line B 6-panel grid, depth-colored markers | Qualitative — Line B |
        | `pointcloud_examples_interactive_v2/dropdown/lineA_<class>_<id>_dropdown.html` | Line A dropdown method switcher | Presentation |
        | `pointcloud_examples_interactive_v2/dropdown/lineB_<class>_<id>_dropdown.html` | Line B dropdown method switcher | Presentation |
        | `pointcloud_examples_interactive_v2/single_method/lineA_<class>_<id>_<method>.html` | Line A single-method detail view | Presentation / supplementary |
        | `pointcloud_examples_interactive_v2/single_method/lineB_<class>_<id>_<method>.html` | Line B single-method detail view | Presentation / supplementary |

        Classes covered: airplane, chair, table, car, sofa (test split).
        """
    ).strip()
    index_path.write_text(existing.rstrip() + "\n\n" + block + "\n", encoding="utf-8")


def update_visualization_summary() -> None:
    path = REPORTS / "modelnet40_visualization_summary.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if "## 10. Interactive point cloud viewers (v2)" in existing:
        existing = existing.split("## 10. Interactive point cloud viewers (v2)")[0].rstrip()

    n_html = len(list(OUT_ROOT.rglob("*.html")))
    block = textwrap.dedent(
        f"""
        ## 10. Interactive point cloud viewers (v2)

        - `{OUT_ROOT}` — {n_html} HTML viewers
        - Plotly 3D scatter with rotate / zoom / pan
        - Line A and Line B multi-panel grids (black + depth color modes)
        - Dropdown switcher pages for presentation
        - Single-method detail pages for airplane and chair
        - **Interactive HTML viewers** are intended for qualitative inspection and presentation
        - **Static figures** in `pointcloud_examples/` remain for thesis document insertion
        - Interactive 3D viewers provide clearer structural comparison than orthographic static PNGs
        - Audit report: `reports/modelnet40_interactive_pointcloud_v2_audit.md`
        """
    ).strip()
    path.write_text(existing + "\n\n" + block + "\n", encoding="utf-8")


def update_captions() -> None:
    path = REPORTS / "modelnet40_thesis_figure_captions.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if "## Interactive point cloud viewers (v2)" in existing:
        existing = existing.split("## Interactive point cloud viewers (v2)")[0].rstrip()

    block = textwrap.dedent(
        f"""
        ## Interactive point cloud viewers (v2)

        **Figure: Line B interactive grid** (`pointcloud_examples_interactive_v2/lineB/lineB_<class>_<id>_interactive_grid_depth.html`)

        Interactive 3D comparison of test-set point clouds under Line B: Original 1024, Downsampled ×4 256, and four upsampling methods at 1024 points. All subplots share identical axis limits and initial camera for fair visual comparison. Black and depth-colored variants are available.

        **Figure: Line A interactive grid** (`pointcloud_examples_interactive_v2/lineA/lineA_<class>_<id>_interactive_grid_depth.html`)

        Interactive 3D comparison under Line A: Original 1024 vs four upsampling methods at 4096 points.

        **Figure: Dropdown viewer** (`pointcloud_examples_interactive_v2/dropdown/`)

        Single-scene presentation viewer with dropdown method selection; camera remains stable when switching methods.

        **Figure: Single-method detail** (`pointcloud_examples_interactive_v2/single_method/`)

        Full-size interactive view for individual methods (airplane and chair). Intended for detailed qualitative inspection during presentation or review.

        Interactive HTML viewers are for qualitative inspection and presentation; static PNG/PDF figures remain for thesis document insertion.
        """
    ).strip()
    path.write_text(existing + "\n\n" + block + "\n", encoding="utf-8")


def main() -> None:
    for sub in ("lineA", "lineB", "dropdown", "single_method"):
        (OUT_ROOT / sub).mkdir(parents=True, exist_ok=True)

    samples = resolve_samples()
    for class_name, sample_id in samples:
        try:
            process_sample_lineb(class_name, sample_id)
        except Exception as exc:
            FAILURES.append(f"Line B {class_name}/{sample_id} unexpected: {exc}")
        try:
            process_sample_linea(class_name, sample_id)
        except Exception as exc:
            FAILURES.append(f"Line A {class_name}/{sample_id} unexpected: {exc}")

    write_audit_report()
    update_figure_index()
    update_visualization_summary()
    update_captions()

    html_count = len([p for p in GENERATED if p.endswith(".html")])
    print(f"Generated {html_count} HTML files under {OUT_ROOT}")
    print(f"Audit: {REPORTS / 'modelnet40_interactive_pointcloud_v2_audit.md'}")
    if FAILURES:
        print("Failures:")
        for f in FAILURES:
            print(" -", f)


if __name__ == "__main__":
    main()
