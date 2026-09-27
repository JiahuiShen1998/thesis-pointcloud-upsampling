#!/usr/bin/env python3
"""Build the final offline thesis visualization bundle."""

from __future__ import annotations

import csv
import html
import json
import os
import re
import shutil
import subprocess
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path("/home/ra87racy/projects/baseline_detectors/PointRCNN")
RESULTS = ROOT / "results"
DEST = RESULTS / "final_visualization_bundle"
ZIP_PATH = RESULTS / "final_visualization_bundle.zip"

PACKAGES = {
    "pointcloud": "interactive_object_crop_visualization_improved_v6",
    "detector": "pointrcnn_detection_box_visualization_v1",
    "failure": "upsampling_detection_failure_explanation_v2",
}

MAIN_METHODS = ["EAR", "PU-Net", "PU-GCN", "PDANS", "TULIP"]
SUPPLEMENTARY_METHODS = ["SPU-PMD"]
SELECTED_FRAMES = ["000093", "000242", "003219", "006833", "007458"]

TEXT_EXTS = {".html", ".md", ".csv", ".json", ".txt", ".js", ".css"}
PLOTLY_RE = re.compile(r"https://cdn\.plot\.ly/(plotly-[0-9.]+\.min\.js)")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def rel_link(from_file: Path, target: Path) -> str:
    return Path(os.path.relpath(target, from_file.parent)).as_posix()


def nice_line(line: str) -> str:
    if line.startswith("line_a"):
        return "Line A: Original baseline vs Original + Upsampling"
    if line.startswith("line_b"):
        return "Line B: Downsampled baseline vs Downsampled + Upsampling"
    return line


def slug_method(method: str) -> str:
    return method.replace("/", "_")


def method_role(method: str) -> str:
    if method in MAIN_METHODS:
        return "main"
    if method in SUPPLEMENTARY_METHODS:
        return "supplementary"
    return "other"


def summary_rel_path(method: str) -> str:
    if method_role(method) == "supplementary":
        return f"supplementary_methods/{slug_method(method)}_summary.html"
    return f"method_summaries/{slug_method(method)}_summary.html"


def method_from_line(line: str) -> str:
    suffix = line.split("_vs_", 1)[-1]
    return suffix


def status_class(status: str) -> str:
    s = status.lower()
    if "ok" in s or "ap-consistent" in s or "complete" in s:
        return "ok"
    if "warning" in s or "qualitative" in s or "partial" in s:
        return "warn"
    if "missing" in s or "error" in s or "incomplete" in s:
        return "bad"
    return "neutral"


CSS = """
:root{--bg:#f7f8fb;--ink:#182030;--muted:#5d6878;--line:#d9dfeb;--blue:#1f5fbf;--green:#247a4d;--amber:#9a6500;--red:#a33a3a;--panel:#ffffff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:Arial,Helvetica,sans-serif;line-height:1.45}
header{background:#162033;color:white;padding:28px 36px}
main{max-width:1240px;margin:0 auto;padding:28px 28px 56px}
h1{margin:0 0 8px;font-size:30px;letter-spacing:0}
h2{margin:30px 0 12px;font-size:21px}
h3{margin:22px 0 8px;font-size:17px}
p{margin:8px 0;color:var(--muted)}
a{color:var(--blue);text-decoration:none}
a:hover{text-decoration:underline}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px}
.compact{font-size:14px}
.badge{display:inline-block;border-radius:999px;padding:3px 8px;margin:2px 4px 2px 0;font-size:12px;border:1px solid var(--line);background:#fff;color:var(--muted)}
.ok{color:var(--green);border-color:#a8d7bd;background:#f0fbf5}
.warn{color:var(--amber);border-color:#e8ce8d;background:#fff8e6}
.bad{color:var(--red);border-color:#e7b1b1;background:#fff0f0}
.neutral{color:#455469;background:#f3f5f9}
table{width:100%;border-collapse:collapse;background:white;border:1px solid var(--line);border-radius:8px;overflow:hidden}
th,td{border-bottom:1px solid var(--line);padding:9px 10px;text-align:left;vertical-align:top;font-size:14px}
th{background:#edf1f7;color:#263246}
tr:last-child td{border-bottom:0}
.nav{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px}
.nav a{background:white;border:1px solid var(--line);border-radius:6px;padding:7px 10px}
code{background:#edf1f7;padding:1px 4px;border-radius:4px}
ul{padding-left:20px}
""".strip()


def html_page(title: str, body: str, rel_to_root: str = ".") -> str:
    css_href = f"{rel_to_root}/assets/bundle.css" if rel_to_root != "." else "assets/bundle.css"
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><link rel="stylesheet" href="{css_href}"></head>
<body><header><h1>{html.escape(title)}</h1><p>Final offline thesis visualization bundle for PointRCNN upsampling analysis.</p></header>
<main>{body}</main></body></html>
"""


def link(label: str, target: str) -> str:
    return f'<a href="{html.escape(target)}">{html.escape(label)}</a>'


def summarize_inputs() -> dict:
    pc_rows = read_csv(DEST / PACKAGES["pointcloud"] / "summary_all.csv")
    det_rows = read_csv(DEST / PACKAGES["detector"] / "summary_all.csv")
    fail_rows = read_csv(DEST / PACKAGES["failure"] / "case_level_failure_analysis.csv")
    ml_rows = read_csv(DEST / PACKAGES["failure"] / "method_line_summary.csv")

    methods = set()
    lines_by_method: dict[str, set[str]] = defaultdict(set)
    for key, package in PACKAGES.items():
        pkg_dir = DEST / package
        if pkg_dir.exists():
            for p in pkg_dir.iterdir():
                if p.is_dir():
                    methods.add(p.name)
    for rows in (pc_rows, det_rows, fail_rows, ml_rows):
        for r in rows:
            m = r.get("method", "")
            line = r.get("line", "")
            if m:
                methods.add(m)
            if m and line:
                lines_by_method[m].add(line)

    html_counts = {
        key: len(list((DEST / package).rglob("*.html"))) if (DEST / package).exists() else 0
        for key, package in PACKAGES.items()
    }

    package_counts = defaultdict(lambda: defaultdict(Counter))
    for key, package in PACKAGES.items():
        pkg_dir = DEST / package
        if not pkg_dir.exists():
            continue
        for method in methods:
            for line_dir in (pkg_dir / method).glob("*"):
                if line_dir.is_dir():
                    package_counts[method][line_dir.name][key] += len(list(line_dir.rglob("*.html")))

    fail_by_ml = {(r["method"], r["line"]): r for r in ml_rows if r.get("method") and r.get("line")}
    det_by_ml = defaultdict(list)
    for r in det_rows:
        det_by_ml[(r.get("method", ""), r.get("line", ""))].append(r)
    pc_by_ml = defaultdict(list)
    for r in pc_rows:
        pc_by_ml[(r.get("method", ""), r.get("line", ""))].append(r)
    fail_cases_by_ml = defaultdict(list)
    for r in fail_rows:
        fail_cases_by_ml[(r.get("method", ""), r.get("line", ""))].append(r)

    return {
        "pc_rows": pc_rows,
        "det_rows": det_rows,
        "fail_rows": fail_rows,
        "ml_rows": ml_rows,
        "methods": sorted(methods),
        "lines_by_method": {m: sorted(v) for m, v in lines_by_method.items()},
        "html_counts": html_counts,
        "package_counts": package_counts,
        "fail_by_ml": fail_by_ml,
        "det_by_ml": det_by_ml,
        "pc_by_ml": pc_by_ml,
        "fail_cases_by_ml": fail_cases_by_ml,
    }


def copy_sources() -> None:
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)
    for package in PACKAGES.values():
        src = RESULTS / package
        shutil.copytree(src, DEST / package)
    (DEST / "assets").mkdir()
    write_text(DEST / "assets" / "bundle.css", CSS + "\n")


def ensure_plotly() -> None:
    for filename in ("plotly-2.32.0.min.js", "plotly-2.35.2.min.js"):
        target = DEST / "assets" / filename
        if target.exists():
            continue
        try:
            urllib.request.urlretrieve(f"https://cdn.plot.ly/{filename}", target)
        except Exception:
            write_text(target, "/* Plotly download failed; original CDN links were replaced for offline packaging. */\n")


def rewrite_text_files() -> None:
    project_prefix = str(ROOT) + "/"
    results_prefix = str(RESULTS) + "/"
    for path in DEST.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTS:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        original = text
        if "https://cdn.plot.ly/" in text:
            text = PLOTLY_RE.sub(lambda m: rel_link(path, DEST / "assets" / m.group(1)), text)
        if results_prefix in text:
            def repl_results(match: re.Match[str]) -> str:
                tail = match.group(1)
                src_target = DEST / tail
                return rel_link(path, src_target)
            text = re.sub(re.escape(results_prefix) + r"(interactive_object_crop_visualization_improved_v6|pointrcnn_detection_box_visualization_v1|upsampling_detection_failure_explanation_v2)([A-Za-z0-9_./\-]*)",
                          lambda m: repl_results(type("M", (), {"group": lambda _self, i: m.group(1) + m.group(2)})()), text)
        text = text.replace(project_prefix, "PROJECT_ROOT/")
        if text != original:
            path.write_text(text, encoding="utf-8")


def first_existing(base: Path, candidates: list[str]) -> str:
    for c in candidates:
        p = base / c
        if p.exists():
            return c
    return ""


def case_links(case: dict[str, str], from_file: Path, package: str = "failure") -> dict[str, str]:
    base = DEST / PACKAGES[package]
    out = {}
    for key in ("dashboard", "evaluation_fov_annotated", "pointcloud_change_view", "detector_change_view"):
        val = case.get(key, "")
        if val:
            out[key] = rel_link(from_file, base / val)
    return out


def build_method_pages(data: dict) -> None:
    out_dir = DEST / "method_summaries"
    supp_dir = DEST / "supplementary_methods"
    out_dir.mkdir(exist_ok=True)
    supp_dir.mkdir(exist_ok=True)
    rows_overview = []
    for method in data["methods"]:
        role = method_role(method)
        page = DEST / summary_rel_path(method)
        lines = data["lines_by_method"].get(method, [])
        if not lines:
            for pkg in PACKAGES.values():
                method_dir = DEST / pkg / method
                if method_dir.exists():
                    lines.extend([p.name for p in method_dir.iterdir() if p.is_dir()])
            lines = sorted(set(lines))

        body = [f'<div class="nav">{link("Home", "../index.html")} {link("Thesis navigation", "../thesis_navigation.html")}</div>']
        body.append("<h2>Method Identity</h2>")
        body.append(f"<p><strong>Method:</strong> {html.escape(method)}</p>")
        if role == "supplementary":
            body.append("<p><span class='badge warn'>supplementary</span><span class='badge warn'>feasibility-only</span><span class='badge warn'>qualitative-only</span> This method is not part of the main AP-consistent method comparison.</p>")
        else:
            body.append("<p><span class='badge ok'>main method</span> This method belongs to the main thesis comparison set.</p>")
        body.append("<table><tr><th>Line</th><th>Baseline comparison</th><th>Coverage</th><th>Provenance</th><th>Interpretation</th></tr>")
        for line in lines:
            pc_n = data["package_counts"][method][line]["pointcloud"]
            det_n = data["package_counts"][method][line]["detector"]
            fail_n = data["package_counts"][method][line]["failure"]
            ml = data["fail_by_ml"].get((method, line), {})
            dets = data["det_by_ml"].get((method, line), [])
            verdicts = Counter(d.get("ap_consistency_verdict", "") for d in dets)
            if ml.get("interpretation"):
                interp = ml["interpretation"]
            elif pc_n and not det_n and not fail_n:
                interp = "Point-cloud evidence is available, but detector and failure-analysis content is missing for this method/line."
            else:
                interp = "Inspect linked dashboards for details."
            provenance = ml.get("dominant_failure_mode") or "not summarized"
            ap_status = "AP-consistent" if verdicts.get("OK", 0) and not (verdicts.get("WARNING", 0) or verdicts.get("ERROR", 0)) else ("provenance warning" if dets else "missing provenance")
            body.append(f"<tr><td>{html.escape(line)}</td><td>{html.escape(nice_line(line))}</td><td>point cloud {pc_n}, detector {det_n}, failure {fail_n}</td><td><span class='badge {status_class(ap_status)}'>{html.escape(ap_status)}</span></td><td>{html.escape(interp)}</td></tr>")
            rows_overview.append([method, role, line, pc_n, det_n, fail_n, ap_status, provenance])
        body.append("</table>")

        for line in lines:
            body.append(f"<h2>{html.escape(nice_line(line))}</h2>")
            pc_dir = DEST / PACKAGES["pointcloud"] / method / line
            det_dir = DEST / PACKAGES["detector"] / method / line
            fail_dir = DEST / PACKAGES["failure"] / method / line

            body.append("<div class='grid'>")
            for label, pkg_key, d in [("Point-Cloud Section", "pointcloud", pc_dir), ("Detector Section", "detector", det_dir), ("Failure Explanation Section", "failure", fail_dir)]:
                body.append("<div class='card'>")
                body.append(f"<h3>{label}</h3>")
                if d.exists():
                    idx = first_existing(d, ["index.html"])
                    frame_dash = sorted(d.glob("frame_*/combined_dashboard.html"))
                    analysis_dash = sorted(d.glob("frame_*/analysis_dashboard.html"))
                    full = sorted(d.glob("frame_*/full_frame.html")) + sorted(d.glob("frame_*/baseline_full_frame.html")) + sorted(d.glob("frame_*/full_lidar_context_annotated.html"))
                    crop = sorted(d.glob("frame_*/object_crop.html")) + sorted(d.glob("frame_*/object_box_*.html"))
                    links = []
                    if idx:
                        links.append(link("line index", rel_link(page, d / idx)))
                    if frame_dash:
                        links.append(link("dashboard", rel_link(page, frame_dash[0])))
                    if analysis_dash:
                        links.append(link("analysis dashboard", rel_link(page, analysis_dash[0])))
                    if full:
                        links.append(link("full-frame / FOV view", rel_link(page, full[0])))
                    if crop:
                        links.append(link("object crop / object page", rel_link(page, crop[0])))
                    links.append(link("folder", rel_link(page, d)))
                    body.append("<p>" + " | ".join(links) + "</p>")
                    body.append(f"<p class='compact'>{len(list(d.rglob('*.html')))} HTML pages available.</p>")
                else:
                    body.append("<p><span class='badge bad'>missing inputs</span> No copied content exists for this package and line.</p>")
                body.append("</div>")
            body.append("</div>")

            cases = data["fail_cases_by_ml"].get((method, line), [])
            if cases:
                body.append("<h3>Representative Examples</h3><table><tr><th>Frame</th><th>Failure mode</th><th>AP status</th><th>Evidence links</th></tr>")
                for c in cases[:6]:
                    links = case_links(c, page)
                    evidence = " | ".join(link(k.replace("_", " "), v) for k, v in links.items())
                    body.append(f"<tr><td>{html.escape(c.get('frame_id',''))}</td><td>{html.escape(c.get('dominant_failure_mode',''))}</td><td><span class='badge {status_class(c.get('ap_provenance_status',''))}'>{html.escape(c.get('ap_provenance_status',''))}</span></td><td>{evidence}</td></tr>")
                body.append("</table>")

        write_text(page, html_page(f"{method} Method Summary", "\n".join(body), ".."))

    with (DEST / "method_overview.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "role", "line", "package_availability", "AP provenance status", "number of point-cloud pages", "number of detector pages", "number of failure-analysis pages", "dominant failure mode", "representative frame links or IDs"])
        for method, role, line, pc_n, det_n, fail_n, ap_status, provenance in rows_overview:
            cases = data["fail_cases_by_ml"].get((method, line), [])
            frames = ";".join(c.get("frame_id", "") for c in cases[:5])
            availability = f"pointcloud={pc_n>0};detector={det_n>0};failure={fail_n>0}"
            w.writerow([method, role, line, availability, ap_status, pc_n, det_n, fail_n, provenance, frames])


def build_package_manifest(data: dict) -> None:
    with (DEST / "package_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file path", "package type", "method", "line", "frame if applicable", "status", "provenance label"])
        for pkg_key, package in PACKAGES.items():
            for p in sorted((DEST / package).rglob("*")):
                if not p.is_file():
                    continue
                rel = p.relative_to(DEST).as_posix()
                parts = p.relative_to(DEST / package).parts
                method = parts[0] if len(parts) > 0 else ""
                line = parts[1] if len(parts) > 1 else ""
                frame = next((x.replace("frame_", "") for x in parts if x.startswith("frame_")), "")
                status = "complete" if p.exists() else "missing"
                prov = ""
                if pkg_key == "detector" and line:
                    rows = data["det_by_ml"].get((method, line), [])
                    verdicts = Counter(r.get("ap_consistency_verdict", "") for r in rows)
                    prov = "AP-consistent" if verdicts.get("OK", 0) and not verdicts.get("WARNING", 0) and not verdicts.get("ERROR", 0) else ("provenance warning" if rows else "")
                w.writerow([rel, pkg_key, method, line, frame, status, prov])


def build_index_and_navigation(data: dict) -> None:
    main_methods = [m for m in MAIN_METHODS if m in data["methods"]]
    supplementary_methods = [m for m in SUPPLEMENTARY_METHODS if m in data["methods"]]
    method_links = " ".join(link(m, summary_rel_path(m)) for m in main_methods)
    supplementary_links = " ".join(link(m, summary_rel_path(m)) for m in supplementary_methods)
    report_links = [
        ("Thesis ready findings", f'{PACKAGES["failure"]}/thesis_ready_findings.md'),
        ("Meeting ready examples", f'{PACKAGES["failure"]}/meeting_ready_examples.md'),
        ("Bundle summary", "bundle_summary.md"),
        ("Provenance summary", "provenance_summary.md"),
        ("PU-Net completion audit", "pu_net_completion_audit.md"),
        ("Package manifest", "package_manifest.csv"),
        ("Method overview", "method_overview.csv"),
    ]
    body = f"""
<section class="card"><h2>Purpose</h2>
<p>This is the final integrated offline visualization bundle. It combines point-cloud visualization, PointRCNN detector-output visualization, and root-cause failure explanation to support the thesis question: why does point cloud upsampling not improve PointRCNN detection performance?</p>
<p><strong>Line A</strong> compares Original baseline vs Original + Upsampling. <strong>Line B</strong> compares Downsampled baseline vs Downsampled + Upsampling. Line B is not described as recovery to original performance.</p></section>
<h2>Included Packages</h2><div class="grid">
<div class="card"><h3>Point-cloud visualization v6</h3><p>Full-frame and object-crop evidence.</p><p>{link("Open package", PACKAGES["pointcloud"] + "/index.html")}</p></div>
<div class="card"><h3>Detector-result visualization v1</h3><p>GT boxes, predictions, TP/FP/FN, IoU, score, FOV and full LiDAR context.</p><p>{link("Open package", PACKAGES["detector"] + "/index.html")}</p></div>
<div class="card"><h3>Failure explanation v2</h3><p>Detector evidence, point-cloud evidence, failure categories, object-level explanation pages.</p><p>{link("Open package", PACKAGES["failure"] + "/index.html")}</p></div>
</div>
<h2>Main Methods</h2><p>These are the thesis mainline methods: EAR, PU-Net, PU-GCN, PDANS, and TULIP.</p><div class="nav">{method_links}</div>
<h2>Supplementary Methods</h2><p>SPU-PMD is supplementary, feasibility-only, qualitative-only evidence and is not part of the main AP-consistent comparison.</p><div class="nav">{supplementary_links} {link("Supplementary overview", "comparison_views/supplementary_overview.html")}</div>
<h2>Navigation By Comparison Line</h2><div class="nav">{link("Line A overview", "comparison_views/line_a_overview.html")} {link("Line B overview", "comparison_views/line_b_overview.html")}</div>
<h2>Navigation By Analysis Type</h2><div class="nav">{link("Point-cloud-level analysis", "comparison_views/pointcloud_overview.html")} {link("Detector-level analysis", "comparison_views/detector_overview.html")} {link("Failure / root-cause analysis", "comparison_views/failure_analysis_overview.html")}</div>
<h2>Thesis And Reports</h2><div class="nav">{' '.join(link(a,b) for a,b in report_links)} {link("Thesis navigation", "thesis_navigation.html")} {link("Meeting navigation", "meeting_navigation.md")}</div>
<h2>Status Legend</h2><p><span class="badge ok">AP-consistent</span><span class="badge warn">qualitative-only</span><span class="badge warn">provenance warning</span><span class="badge bad">missing inputs</span><span class="badge ok">complete</span><span class="badge warn">partial</span></p>
"""
    write_text(DEST / "index.html", html_page("Final Visualization Bundle", body))

    nav_sections = [
        ("Does upsampling help on original input? (Line A)", "comparison_views/line_a_overview.html"),
        ("Does upsampling help on downsampled input? (Line B)", "comparison_views/line_b_overview.html"),
        ("What changes in the point cloud?", "comparison_views/pointcloud_overview.html"),
        ("What changes in detector outputs?", "comparison_views/detector_overview.html"),
        ("Why does performance not improve?", "comparison_views/failure_analysis_overview.html"),
    ]
    body = "<div class='nav'>" + link("Home", "index.html") + "</div>"
    body += "<p>Use this page by thesis question rather than by raw file location.</p><div class='grid'>"
    for title, href in nav_sections:
        body += f"<div class='card'><h2>{html.escape(title)}</h2><p>{link('Open overview', href)}</p></div>"
    body += "</div><h2>Failure Mode Routes</h2><table><tr><th>Question</th><th>Recommended path</th></tr>"
    for mode in ["Recall degradation", "Precision degradation", "Localization degradation", "Confidence degradation"]:
        matching = [r for r in data["ml_rows"] if r.get("method") in MAIN_METHODS and mode.lower().split()[0] in r.get("dominant_failure_mode", "").lower()]
        links = []
        for r in matching[:8]:
            links.append(link(f"{r['method']} {r['line']}", summary_rel_path(r["method"])))
        body += f"<tr><td>Which methods show mainly {mode.lower()}?</td><td>{' | '.join(links) or 'No dominant method-level label found.'}</td></tr>"
    body += f"<tr><td>Meeting-ready examples</td><td>{link('Meeting navigation', 'meeting_navigation.md')} | {link('source examples', PACKAGES['failure'] + '/meeting_ready_examples.md')}</td></tr>"
    body += f"<tr><td>Thesis-ready findings</td><td>{link('Thesis findings', PACKAGES['failure'] + '/thesis_ready_findings.md')}</td></tr></table>"
    write_text(DEST / "thesis_navigation.html", html_page("Thesis Navigation", body))

    md = [
        "# Thesis Navigation",
        "",
        "Line A: Original baseline vs Original + Upsampling.",
        "Line B: Downsampled baseline vs Downsampled + Upsampling. Line B is not recovery to original performance.",
        "",
        "Main methods: EAR, PU-Net, PU-GCN, PDANS, TULIP.",
        "Supplementary method: SPU-PMD, feasibility-only qualitative evidence, not part of the main AP-consistent method comparison.",
        "",
    ]
    for title, href in nav_sections:
        md.append(f"- [{title}]({href})")
    md += [
        "",
        "## Failure Mode Routes",
    ]
    for mode in ["Recall degradation", "Precision degradation", "Localization degradation", "Confidence degradation"]:
        md.append(f"- {mode}: see `comparison_views/failure_analysis_overview.html` and method summaries.")
    write_text(DEST / "thesis_navigation.md", "\n".join(md) + "\n")


def build_comparison_pages(data: dict) -> None:
    out = DEST / "comparison_views"
    out.mkdir(exist_ok=True)

    def overview(line_prefix: str, title: str) -> None:
        rows = []
        for m in [x for x in MAIN_METHODS if x in data["methods"]]:
            for line in data["lines_by_method"].get(m, []):
                if line.startswith(line_prefix):
                    ml = data["fail_by_ml"].get((m, line), {})
                    rows.append((m, line, ml))
        body = f"<div class='nav'>{link('Home','../index.html')} {link('Thesis navigation','../thesis_navigation.html')}</div><p>{html.escape(nice_line(line_prefix))}</p>"
        body += "<table><tr><th>Method</th><th>Line</th><th>Dominant finding</th><th>Links</th></tr>"
        for m, line, ml in rows:
            body += f"<tr><td>{html.escape(m)}</td><td>{html.escape(line)}</td><td>{html.escape(ml.get('interpretation','Inspect package evidence.'))}</td><td>{link('method summary', '../' + summary_rel_path(m))}</td></tr>"
        body += "</table>"
        write_text(out / f"{line_prefix}_overview.html", html_page(title, body, ".."))

    overview("line_a", "Line A Overview")
    overview("line_b", "Line B Overview")

    for pkg_key, title, descr in [
        ("pointcloud", "Point-Cloud Evidence Overview", "Visual evidence across full-frame and object-crop views."),
        ("detector", "Detector Evidence Overview", "GT, predictions, TP/FP/FN, IoU, confidence, FOV and full LiDAR context."),
        ("failure", "Failure Analysis Overview", "Dominant reasons why upsampling does not improve detection performance."),
    ]:
        body = f"<div class='nav'>{link('Home','../index.html')}</div><p>{html.escape(descr)}</p>"
        body += "<table><tr><th>Method</th><th>Line</th><th>Pages</th><th>Interpretation</th><th>Links</th></tr>"
        for m in [x for x in MAIN_METHODS if x in data["methods"]]:
            lines = data["lines_by_method"].get(m, [])
            if not lines:
                lines = sorted({p.name for p in (DEST / PACKAGES[pkg_key] / m).glob("*") if p.is_dir()}) if (DEST / PACKAGES[pkg_key] / m).exists() else []
            for line in lines:
                count = data["package_counts"][m][line][pkg_key]
                if not count:
                    continue
                ml = data["fail_by_ml"].get((m, line), {})
                pkg_line = DEST / PACKAGES[pkg_key] / m / line
                pkg_link = rel_link(out / f"{pkg_key}_overview.html", pkg_line)
                body += f"<tr><td>{html.escape(m)}</td><td>{html.escape(line)}</td><td>{count}</td><td>{html.escape(ml.get('dominant_failure_mode','package evidence available'))}</td><td>{link('package line folder', pkg_link)} | {link('method summary', '../' + summary_rel_path(m))}</td></tr>"
        body += "</table>"
        filename = {"pointcloud": "pointcloud_overview.html", "detector": "detector_overview.html", "failure": "failure_analysis_overview.html"}[pkg_key]
        write_text(out / filename, html_page(title, body, ".."))

    body = f"<div class='nav'>{link('Home','../index.html')}</div>"
    body += "<p>Supplementary methods are feasibility-only qualitative evidence and are not part of the main AP-consistent comparison.</p>"
    body += "<table><tr><th>Method</th><th>Line</th><th>Available evidence</th><th>Links</th></tr>"
    for m in [x for x in SUPPLEMENTARY_METHODS if x in data["methods"]]:
        lines = data["lines_by_method"].get(m, [])
        if not lines:
            lines = sorted({p.name for p in (DEST / PACKAGES["pointcloud"] / m).glob("*") if p.is_dir()}) if (DEST / PACKAGES["pointcloud"] / m).exists() else []
        for line in lines:
            pc_n = data["package_counts"][m][line]["pointcloud"]
            det_n = data["package_counts"][m][line]["detector"]
            fail_n = data["package_counts"][m][line]["failure"]
            body += f"<tr><td>{html.escape(m)}</td><td>{html.escape(line)}</td><td>point cloud {pc_n}, detector {det_n}, failure {fail_n}; qualitative-only</td><td>{link('supplementary summary', '../' + summary_rel_path(m))}</td></tr>"
    body += "</table>"
    write_text(out / "supplementary_overview.html", html_page("Supplementary Methods Overview", body, ".."))


def rel_project(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def count_files(path: Path, pattern: str) -> int:
    return len(list(path.glob(pattern))) if path.exists() else 0


def build_punet_completion_audit(data: dict) -> dict:
    mappings = {
        "line_a_original_vs_punet": {
            "base_cloud": ROOT / "data/KITTI/object/training/velodyne_original_val",
            "up_cloud": ROOT / "data/KITTI/object/training/velodyne_punet_x2_fullframe",
            "base_det": ROOT / "results/pointrcnn_clean_original_baseline_20260508_203809/evaluation/raw_output/eval/epoch_no_number/val/final_result/data",
            "up_det": ROOT / "results/pointrcnn_50frame_default_validation_cap100k/inference/punet_default/eval/epoch_no_number/val_pugcn_cap100k_50/test_mode/final_result/data",
        },
        "line_b_downsampled_vs_punet": {
            "base_cloud": ROOT / "data/KITTI/object/training/velodyne_downsampled_50_val",
            "up_cloud": ROOT / "data/KITTI/object/training/velodyne_downsampled_50_punet_x2_fullframe",
            "base_det": ROOT / "results/rpn4096_no_distance_propose_comparison/01_downsampled50/inference/eval/epoch_no_number/val/test_mode/final_result/data",
            "up_det": ROOT / "results/rpn4096_no_distance_propose_comparison/03_downsampled50_PUNet/inference/eval/epoch_no_number/val/test_mode/final_result/data",
        },
    }
    rows = []
    line_status = {}
    for line, mp in mappings.items():
        pc_pages = data["package_counts"]["PU-Net"][line]["pointcloud"]
        det_pages = data["package_counts"]["PU-Net"][line]["detector"]
        fail_pages = data["package_counts"]["PU-Net"][line]["failure"]
        det_rows = data["det_by_ml"].get(("PU-Net", line), [])
        fail_cases = data["fail_cases_by_ml"].get(("PU-Net", line), [])
        selected_missing = []
        selected_present = []
        for frame in SELECTED_FRAMES:
            exists = (mp["up_det"] / f"{frame}.txt").exists()
            (selected_present if exists else selected_missing).append(frame)
        up_det_count = count_files(mp["up_det"], "*.txt")
        base_det_count = count_files(mp["base_det"], "*.txt")
        up_cloud_count = count_files(mp["up_cloud"], "*.bin")
        base_cloud_count = count_files(mp["base_cloud"], "*.bin")
        ap_verdicts = Counter(r.get("ap_consistency_verdict", "") for r in det_rows)
        if not mp["up_det"].exists():
            missing_reason = "missing detection txt folder"
        elif selected_missing:
            missing_reason = "missing detection txt for selected v1/v2 frames"
        elif ap_verdicts.get("WARNING") or ap_verdicts.get("ERROR"):
            missing_reason = "detector output exists but has AP provenance warning/error"
        else:
            missing_reason = "none"
        if det_pages and fail_pages and not selected_missing and not (ap_verdicts.get("WARNING") or ap_verdicts.get("ERROR")):
            status = "complete"
        elif det_pages or fail_pages:
            status = "partial detector/failure coverage"
        else:
            status = "point-cloud complete; detector/failure missing"
        line_status[line] = status
        rows.append({
            "method": "PU-Net",
            "line": line,
            "role": "main",
            "point_cloud_base_folder": rel_project(mp["base_cloud"]),
            "point_cloud_upsampled_folder": rel_project(mp["up_cloud"]),
            "base_point_cloud_count": base_cloud_count,
            "upsampled_point_cloud_count": up_cloud_count,
            "v6_pointcloud_html_pages": pc_pages,
            "baseline_detection_folder": rel_project(mp["base_det"]),
            "upsampled_detection_folder": rel_project(mp["up_det"]),
            "baseline_detection_txt_count": base_det_count,
            "upsampled_detection_txt_count": up_det_count,
            "selected_frames_with_upsampled_detection_txt": ";".join(selected_present),
            "selected_frames_missing_upsampled_detection_txt": ";".join(selected_missing),
            "v1_detector_html_pages": det_pages,
            "v2_failure_html_pages": fail_pages,
            "gt_prediction_matching_available": "yes" if det_rows else "no",
            "tp_fp_fn_iou_score_available": "yes" if fail_cases or det_rows else "no",
            "ap_provenance_status": "AP-consistent" if det_rows and ap_verdicts.get("OK") and not (ap_verdicts.get("WARNING") or ap_verdicts.get("ERROR")) else ("provenance warning / selected-frame subset" if det_rows else "missing detector provenance"),
            "completion_status": status,
            "missing_reason": missing_reason,
            "can_be_completed_by_path_mapping": "no, no full/selected matching PU-Net result folder was found at the mapped location" if selected_missing else "not needed",
            "requires_pointrcnn_rerun": "yes" if selected_missing or not det_rows else "no",
        })

    fieldnames = list(rows[0].keys())
    with (DEST / "pu_net_completion_audit.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    md = [
        "# PU-Net Completion Audit",
        "",
        "PU-Net is a main method in the thesis comparison. It must remain in the main method navigation even when detector or failure-analysis coverage is partial.",
        "",
        "## Summary",
        "",
        f"- PU-Net Line A status: {line_status['line_a_original_vs_punet']}",
        f"- PU-Net Line B status: {line_status['line_b_downsampled_vs_punet']}",
        "- Point-cloud inputs exist for both Line A and Line B and the v6 point-cloud visualizations are available.",
        "- Detector coverage is partial: Line A has selected/subset detector output only; Line B does not have detection txt files for the selected v1/v2 frames.",
        "- Failure-analysis coverage follows detector availability: Line A has one frame of failure analysis; Line B has no PU-Net failure-analysis pages.",
        "",
        "## Detailed Status",
        "",
        "| Line | Point cloud | Detector txt | v1 detector pages | v2 failure pages | AP/provenance | Missing reason |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for row in rows:
        pc = f"{row['base_point_cloud_count']} baseline / {row['upsampled_point_cloud_count']} upsampled bins; {row['v6_pointcloud_html_pages']} v6 pages"
        det = f"{row['baseline_detection_txt_count']} baseline / {row['upsampled_detection_txt_count']} upsampled txt; missing selected frames: {row['selected_frames_missing_upsampled_detection_txt'] or 'none'}"
        md.append(f"| {row['line']} | {pc} | {det} | {row['v1_detector_html_pages']} | {row['v2_failure_html_pages']} | {row['ap_provenance_status']} | {row['missing_reason']} |")
    md += [
        "",
        "## Can Existing Result Folders Complete PU-Net?",
        "",
        "The mapped PU-Net detector folders exist only as subset outputs. The Line A mapped folder contains 50 txt files and covers frame `000093`, but not the other selected explanation frames. The Line B mapped folder contains 14 txt files and does not cover the selected explanation frames. No full AP-consistent PU-Net result folder was identified in the focused local scan.",
        "",
        "Therefore, PU-Net cannot be made complete by a simple path-mapping fix with the currently identified files. It requires a PointRCNN evaluation rerun, or locating an unindexed full PU-Net evaluation folder that contains the missing selected frames and full validation provenance.",
        "",
        "## Rerun Plan",
        "",
        "Use the same PointRCNN settings as the existing main comparisons:",
        "",
        "- Config: `cfgs/default.yaml`",
        "- Checkpoint: `PointRCNN.pth`",
        "- Evaluation mode: `--eval_mode rcnn --save_result --test`",
        "- Common override: `RPN.LOC_XZ_FINE False`",
        "- Use the full validation split for AP consistency, not a smoke or selected-frame split.",
        "",
        "Line A required input:",
        "",
        "- Baseline cloud folder: `data/KITTI/object/training/velodyne_original_val`",
        "- PU-Net upsampled cloud folder: `data/KITTI/object/training/velodyne_punet_x2_fullframe`",
        "- Expected output: a full-validation `final_result/data` folder under a clearly named PU-Net Line A result directory.",
        "",
        "Line B required input:",
        "",
        "- Baseline cloud folder: `data/KITTI/object/training/velodyne_downsampled_50_val`",
        "- PU-Net upsampled cloud folder: `data/KITTI/object/training/velodyne_downsampled_50_punet_x2_fullframe`",
        "- Expected output: a full-validation `final_result/data` folder under a clearly named PU-Net Line B result directory.",
        "",
        "After rerun, update the detector path mapping in `tools/generate_pointrcnn_detection_box_visualization_v1.py`, regenerate v1 detector visualization, regenerate v2 failure analysis, and rebuild this final bundle.",
    ]
    write_text(DEST / "pu_net_completion_audit.md", "\n".join(md) + "\n")
    return {
        "line_a_complete": line_status["line_a_original_vs_punet"] == "complete",
        "line_b_complete": line_status["line_b_downsampled_vs_punet"] == "complete",
        "line_a_status": line_status["line_a_original_vs_punet"],
        "line_b_status": line_status["line_b_downsampled_vs_punet"],
    }


def build_reports(data: dict, validation: dict | None = None, punet_audit: dict | None = None) -> None:
    html_total = len(list(DEST.rglob("*.html")))
    report_files = [p for p in DEST.rglob("*") if p.is_file() and p.suffix.lower() in {".md", ".csv", ".json", ".txt"}]
    main_rows = [r for r in data["fail_rows"] if r.get("method") in MAIN_METHODS]
    ap_cases = sum(1 for r in main_rows if r.get("ap_provenance_status") == "AP-consistent")
    qualitative_cases = len(main_rows) - ap_cases
    modes = Counter(r.get("dominant_failure_mode", "unspecified") for r in main_rows)

    summary = [
        "# Bundle Summary",
        "",
        f"- Main methods: {', '.join([m for m in MAIN_METHODS if m in data['methods']])}",
        f"- Supplementary methods: {', '.join([m for m in SUPPLEMENTARY_METHODS if m in data['methods']]) or 'none'}",
        f"- Total HTML files: {html_total}",
        f"- Report and summary files: {len(report_files)}",
        f"- Package breakdown: point-cloud {data['html_counts']['pointcloud']} HTML, detector {data['html_counts']['detector']} HTML, failure analysis {data['html_counts']['failure']} HTML",
        f"- Main-method AP-consistent integrated cases: {ap_cases}",
        f"- Main-method qualitative/provenance-warning integrated cases: {qualitative_cases}",
    ]
    if punet_audit:
        summary += [
            f"- PU-Net Line A: {punet_audit['line_a_status']}",
            f"- PU-Net Line B: {punet_audit['line_b_status']}",
        ]
    summary += [
        "",
        "## Major Findings",
        "",
        "Upsampling does not consistently improve PointRCNN detection relative to either comparison baseline. Across selected cases, added point density often fails to add detector-useful object geometry and can coincide with missed detections, false positives, lower IoU, or lower confidence.",
        "",
        "SPU-PMD is retained only as supplementary feasibility evidence and is not part of the main AP-consistent method comparison.",
        "",
        "## Dominant Failure Patterns",
    ]
    for mode, n in modes.most_common():
        summary.append(f"- {mode}: {n} cases")
    if validation:
        summary += ["", "## Validation", "", json.dumps(validation, indent=2)]
    write_text(DEST / "bundle_summary.md", "\n".join(summary) + "\n")

    readme = f"""# Final Visualization Bundle

Open `index.html` in a browser to browse the bundle locally.

This bundle combines:

- `{PACKAGES['pointcloud']}`: point-cloud full-frame and object-crop visualization.
- `{PACKAGES['detector']}`: PointRCNN detector visualization with GT, predictions, TP/FP/FN, IoU, confidence, FOV, and full LiDAR context.
- `{PACKAGES['failure']}`: integrated root-cause and failure explanation.

Line A means Original baseline vs Original + Upsampling.
Line B means Downsampled baseline vs Downsampled + Upsampling. Line B is only a comparison to the downsampled baseline.

Use `thesis_navigation.html` to navigate by thesis question, `method_summaries/` for method-level pages, and `comparison_views/` for Line A, Line B, point-cloud, detector, and failure-analysis overviews.

Main methods are EAR, PU-Net, PU-GCN, PDANS, and TULIP. SPU-PMD is supplementary, feasibility-only qualitative evidence and is browsed from `supplementary_methods/`.
"""
    write_text(DEST / "README.md", readme)

    prov = [
        "# Provenance Summary",
        "",
        "Use AP-consistent cases as primary thesis evidence. Qualitative-only or provenance-warning cases are retained because they are useful for visual explanation, but should be interpreted cautiously.",
        "",
        "| Method | Role | Line | Status | Notes |",
        "|---|---|---|---|---|",
    ]
    for m in data["methods"]:
        for line in data["lines_by_method"].get(m, []):
            cases = data["fail_cases_by_ml"].get((m, line), [])
            statuses = Counter(c.get("ap_provenance_status", "missing provenance") for c in cases)
            if statuses:
                status = ", ".join(f"{k}: {v}" for k, v in statuses.items())
            elif data["package_counts"][m][line]["pointcloud"]:
                status = "qualitative-only / missing detector provenance"
            else:
                status = "missing provenance"
            notes = data["fail_by_ml"].get((m, line), {}).get("interpretation", "")
            if m == "PU-Net":
                notes = (notes + " " if notes else "") + "See `pu_net_completion_audit.md` for the main-method completion audit."
            if method_role(m) == "supplementary":
                status = "supplementary / feasibility-only / qualitative-only"
                notes = "Not part of the main AP-consistent method comparison."
            prov.append(f"| {m} | {method_role(m)} | {line} | {status} | {notes} |")
    write_text(DEST / "provenance_summary.md", "\n".join(prov) + "\n")

    meeting = [
        "# Meeting Navigation",
        "",
        "Open these first:",
        "",
        "- `index.html`",
        "- `thesis_navigation.html`",
        "- `pu_net_completion_audit.md` for PU-Net main-method coverage status",
        f"- `{PACKAGES['failure']}/meeting_ready_examples.md`",
        "",
        "Representative routes:",
        "",
        "- FN increase: see failure package meeting-ready FN case and matching method summary.",
        "- FP increase: see failure package meeting-ready FP case and matching method summary.",
        "- IoU drop: see failure package meeting-ready IoU case and matching method summary.",
        "- Score drop: see failure package meeting-ready score-drop case and matching method summary.",
        "- Point-cloud change without detector improvement: compare point-cloud overview with detector and failure overviews.",
        "",
        "SPU-PMD is in `supplementary_methods/` and `comparison_views/supplementary_overview.html`; do not present it as a main AP-consistent method.",
    ]
    write_text(DEST / "meeting_navigation.md", "\n".join(meeting) + "\n")


def validate_links() -> dict:
    link_re = re.compile(r"""(?:href|src)=["']([^"'#]+)(?:#[^"']*)?["']""", re.I)
    checked = 0
    broken = []
    external = []
    for p in DEST.rglob("*.html"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        for raw in link_re.findall(text):
            if raw.startswith(("javascript:", "mailto:", "data:")):
                continue
            parsed = urlparse(raw)
            if parsed.scheme in {"http", "https"}:
                external.append((p.relative_to(DEST).as_posix(), raw))
                continue
            target = (p.parent / unquote(parsed.path)).resolve()
            checked += 1
            if not target.exists():
                broken.append((p.relative_to(DEST).as_posix(), raw))
    absolute_hits = []
    for p in DEST.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in TEXT_EXTS:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "/home/" in text:
            absolute_hits.append(p.relative_to(DEST).as_posix())
    return {
        "working_local_links": checked - len(broken),
        "broken_links": len(broken),
        "broken_link_examples": broken[:20],
        "external_links": len(external),
        "external_link_examples": external[:20],
        "absolute_home_references": len(absolute_hits),
        "absolute_home_examples": absolute_hits[:20],
    }


def zip_bundle() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in DEST.rglob("*"):
            if p.is_file():
                zf.write(p, Path("final_visualization_bundle") / p.relative_to(DEST))


def human_size(path: Path) -> str:
    n = int(subprocess.check_output(["du", "-sb", str(path)]).decode().split()[0]) if path.is_dir() else path.stat().st_size
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def main() -> None:
    copy_sources()
    ensure_plotly()
    rewrite_text_files()
    data = summarize_inputs()
    build_method_pages(data)
    build_package_manifest(data)
    build_index_and_navigation(data)
    build_comparison_pages(data)
    punet_audit = build_punet_completion_audit(data)
    validation = validate_links()
    build_reports(data, validation, punet_audit)
    validation = validate_links()
    build_reports(data, validation, punet_audit)
    zip_bundle()

    final_report = {
        "bundle_path": str(DEST),
        "zip_path": str(ZIP_PATH),
        "main_methods_included": len([m for m in MAIN_METHODS if m in data["methods"]]),
        "main_methods": [m for m in MAIN_METHODS if m in data["methods"]],
        "supplementary_methods": [m for m in SUPPLEMENTARY_METHODS if m in data["methods"]],
        "punet_completion": punet_audit,
        "total_html_files": len(list(DEST.rglob("*.html"))),
        "report_summary_files": len([p for p in DEST.rglob("*") if p.is_file() and p.suffix.lower() in {".md", ".csv", ".json", ".txt"}]),
        "validation": validation,
        "bundle_size": human_size(DEST),
        "zip_size": human_size(ZIP_PATH),
        "subpackage_html_counts": data["html_counts"],
        "main_methods_with_partial_or_missing_detector_failure_coverage": [
            m for m in [x for x in MAIN_METHODS if x in data["methods"]]
            if any(data["package_counts"][m][line]["detector"] == 0 or data["package_counts"][m][line]["failure"] == 0 for line in data["lines_by_method"].get(m, []))
        ],
    }
    write_text(DEST / "validation_report.json", json.dumps(final_report, indent=2) + "\n")
    print(json.dumps(final_report, indent=2))


if __name__ == "__main__":
    main()
