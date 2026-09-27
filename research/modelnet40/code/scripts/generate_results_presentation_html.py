#!/usr/bin/env python3
"""Generate a browser-based HTML slide deck for presenting ModelNet40 results."""

from __future__ import annotations

from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
FIG = PROJECT / "figures" / "modelnet40"
PKG = FIG / "final_visualization_package"
OUT = PROJECT / "presentations" / "ModelNet40_Results_Presentation.html"


def rel(p: Path) -> str:
    return str(p.relative_to(PROJECT)).replace("\\", "/")


slides: list[tuple[str, str]] = []


def add(title: str, body: str) -> None:
    slides.append((title, body))


add(
    "Point Cloud Upsampling on ModelNet40",
    """
    <p class="subtitle">Geometry · Qualitative Point Clouds · PointNet++ Classification</p>
    <ul>
      <li>Line A: Original 1024 → upsample to 4096</li>
      <li>Line B: Downsample ×4 (256) → upsample to 1024</li>
      <li>Methods: EAR · PDANS · PU-Net · PU-GCN · PU-EdgeFormer</li>
    </ul>
    """,
)

add(
    "Three Core Comparisons",
    """
    <ol>
      <li><b>Geometry quality</b> — CD / HD / NUC / exact P2F vs dense mesh</li>
      <li><b>Line A classification</b> — does densification help PointNet++?</li>
      <li><b>Line B classification</b> — can upsampling recover after ×4 downsample?</li>
    </ol>
    <p class="note">Key theme: geometry best ≠ classification best.</p>
    """,
)

add(
    "Line B Geometry Table (summary)",
    """
    <table>
      <tr><th>Method</th><th>CD</th><th>HD</th><th>NUC</th><th>P2F</th></tr>
      <tr><td>Original 1024</td><td>0.050</td><td>0.120</td><td>1.06</td><td>0.082</td></tr>
      <tr><td>Down ×4 256</td><td>0.077</td><td>0.214</td><td>2.05</td><td>0.082</td></tr>
      <tr><td>EAR</td><td>0.081</td><td>0.209</td><td>0.96</td><td>0.081</td></tr>
      <tr><td>PDANS</td><td>0.063</td><td>0.206</td><td>1.09</td><td>0.080</td></tr>
      <tr><td>PU-Net</td><td>0.057</td><td>0.138</td><td>1.06</td><td>0.080</td></tr>
      <tr><td>PU-GCN</td><td><b>0.055</b></td><td>0.165</td><td>1.49↑</td><td>0.084</td></tr>
      <tr><td>PU-EdgeFormer</td><td>0.063</td><td>0.193</td><td>1.10</td><td>0.084</td></tr>
    </table>
    <p class="note">Lower is better. PU-GCN best CD but high NUC; PU-Net more balanced.</p>
    """,
)

# geometry figures
for name, path in [
    ("Δ CD vs Original", FIG / "geometry_delta_methods_only_with_pu_edgeformer" / "lineB_delta_cd_vs_original_baseline_methods_only_with_pu_edgeformer.png"),
    ("Δ HD vs Original", FIG / "geometry_delta_methods_only_with_pu_edgeformer" / "lineB_delta_hd_vs_original_baseline_methods_only_with_pu_edgeformer.png"),
    ("Δ NUC vs Original", FIG / "geometry_delta_methods_only_with_pu_edgeformer" / "lineB_delta_nuc_vs_original_baseline_methods_only_with_pu_edgeformer.png"),
    ("Δ exact P2F vs Original", FIG / "geometry_delta_methods_only_with_pu_edgeformer" / "lineB_delta_exact_p2f_vs_original_baseline_methods_only_with_pu_edgeformer.png"),
]:
    add(f"Geometry — {name}", f'<img src="../{rel(path)}" alt="{name}">')

# point clouds Line B
for cls, fname in [
    ("Airplane", "lineB_airplane_airplane_0627_comparison.png"),
    ("Chair", "lineB_chair_chair_0890_comparison.png"),
    ("Table", "lineB_table_table_0393_comparison.png"),
    ("Car", "lineB_car_car_0198_comparison.png"),
    ("Sofa", "lineB_sofa_sofa_0681_comparison.png"),
]:
    p = PKG / "pointcloud_static" / "lineB" / fname
    if not p.exists():
        p = FIG / "pointcloud_examples" / fname
    add(f"Line B Point Cloud — {cls}", f'<img src="../{rel(p)}" alt="{cls}">')

for cls, fname in [
    ("Airplane", "lineA_airplane_airplane_0627_comparison.png"),
    ("Chair", "lineA_chair_chair_0890_comparison.png"),
]:
    p = PKG / "pointcloud_static" / "lineA" / fname
    if not p.exists():
        p = FIG / "pointcloud_examples" / fname
    add(f"Line A Point Cloud — {cls}", f'<img src="../{rel(p)}" alt="{cls}">')

add(
    "Line A Classification",
    """
    <table>
      <tr><th>Method</th><th>Best Overall</th><th>Δ vs Original</th></tr>
      <tr><td>Original baseline 1024</td><td><b>91.95%</b></td><td>+0.00 pp</td></tr>
      <tr><td>EAR 4096</td><td>91.48%</td><td>−0.47 pp</td></tr>
      <tr><td>PDANS 4096</td><td>91.62%</td><td>−0.33 pp</td></tr>
      <tr><td>PU-Net 4096</td><td>90.95%</td><td>−1.00 pp</td></tr>
      <tr><td>PU-GCN 4096</td><td>91.63%</td><td>−0.32 pp</td></tr>
      <tr><td>PU-EdgeFormer 4096</td><td>90.54%</td><td>−1.41 pp</td></tr>
    </table>
    <p class="note">All upsampling methods below Original 1024 — densification does not help.</p>
    """,
)

add(
    "Line A Plots",
    f"""
    <div class="row">
      <img src="../{rel(FIG / 'pointnet2_final_with_pu_edgeformer' / 'accuracy_absolute' / 'lineA_best_overall_accuracy_absolute_with_pu_edgeformer.png')}">
      <img src="../{rel(FIG / 'pointnet2_final_with_pu_edgeformer' / 'accuracy_delta_methods_only' / 'lineA_delta_best_overall_vs_original_baseline_methods_only_with_pu_edgeformer.png')}">
    </div>
    """,
)

add(
    "Line B Classification",
    """
    <table>
      <tr><th>Method</th><th>Best Overall</th><th>Δ vs Down ×4</th></tr>
      <tr><td>Downsampled baseline 256</td><td>90.85%</td><td>+0.00 pp</td></tr>
      <tr><td>EAR 1024</td><td>88.81%</td><td>−2.04 pp</td></tr>
      <tr><td>PDANS 1024</td><td>90.34%</td><td>−0.51 pp</td></tr>
      <tr class="hl"><td>PU-Net 1024</td><td><b>91.27%</b></td><td><b>+0.42 pp</b></td></tr>
      <tr><td>PU-GCN 1024</td><td>90.06%</td><td>−0.79 pp</td></tr>
      <tr><td>PU-EdgeFormer 1024</td><td>89.32%</td><td>−1.53 pp</td></tr>
    </table>
    <p class="note">PU-Net is the only method slightly above the downsampled baseline.</p>
    """,
)

add(
    "Line B Plots",
    f"""
    <div class="row">
      <img src="../{rel(FIG / 'pointnet2_final_with_pu_edgeformer' / 'accuracy_absolute' / 'lineB_best_overall_accuracy_absolute_with_pu_edgeformer.png')}">
      <img src="../{rel(FIG / 'pointnet2_final_with_pu_edgeformer' / 'accuracy_delta_methods_only' / 'lineB_delta_best_overall_vs_downsampled_baseline_methods_only_with_pu_edgeformer.png')}">
    </div>
    """,
)

add(
    "Geometry vs Classification (CD / HD)",
    f"""
    <div class="row">
      <img src="../{rel(FIG / 'pointnet2_final_with_pu_edgeformer' / 'final_combined' / 'lineB_delta_cd_vs_delta_accuracy_with_pu_edgeformer.png')}">
      <img src="../{rel(FIG / 'pointnet2_final_with_pu_edgeformer' / 'final_combined' / 'lineB_delta_hd_vs_delta_accuracy_with_pu_edgeformer.png')}">
    </div>
    """,
)

add(
    "Conclusions",
    """
    <ul>
      <li>Upsampling does <b>not</b> universally improve PointNet++ classification.</li>
      <li>Line A: Original 1024 remains strongest (91.95%).</li>
      <li>Line B: only PU-Net gains (+0.42 pp) over downsampled baseline.</li>
      <li>Geometry leader (PU-GCN CD) ≠ classification leader (PU-Net).</li>
      <li>Effect depends on method and downstream task.</li>
    </ul>
    <p class="subtitle">Thank you — Questions?</p>
    """,
)

html_slides = []
for i, (title, body) in enumerate(slides):
    html_slides.append(
        f'<section class="slide" id="s{i}">'
        f"<h1>{title}</h1>{body}"
        f'<div class="footer">{i+1} / {len(slides)} · ← → or click</div>'
        f"</section>"
    )

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>ModelNet40 PointNet++ Results Presentation</title>
<style>
  html,body {{ margin:0; height:100%; background:#0f1720; color:#1a1a1a; font-family: Calibri, Segoe UI, sans-serif; }}
  .deck {{ height:100%; }}
  .slide {{ display:none; box-sizing:border-box; height:100%; padding:2.2vh 4vw 6vh; background:#f7f8fa; }}
  .slide.active {{ display:block; }}
  h1 {{ color:#1a3a5c; font-size: clamp(22px, 3.2vw, 36px); margin:0 0 1.5vh; }}
  .subtitle {{ color:#555; font-size: clamp(16px, 2vw, 22px); }}
  ul,ol {{ font-size: clamp(16px, 2vw, 22px); line-height:1.45; max-width: 1100px; }}
  li {{ margin: 0.55em 0; }}
  table {{ border-collapse:collapse; width:min(1100px, 100%); font-size: clamp(13px, 1.5vw, 18px); background:#fff; }}
  th,td {{ border:1px solid #ccd3db; padding:0.45em 0.6em; text-align:center; }}
  th {{ background:#1a3a5c; color:#fff; }}
  tr.hl td {{ background:#e8f6ea; font-weight:600; }}
  .note {{ color:#444; font-size: clamp(14px, 1.6vw, 18px); margin-top:1em; }}
  img {{ max-width:100%; max-height:78vh; object-fit:contain; display:block; margin:0 auto; background:#fff; }}
  .row {{ display:flex; gap:1vw; align-items:flex-start; }}
  .row img {{ max-width:49%; max-height:75vh; }}
  .footer {{ position:fixed; bottom:12px; right:24px; color:#667; font-size:14px; }}
  .hint {{ position:fixed; bottom:12px; left:24px; color:#889; font-size:13px; }}
</style>
</head>
<body>
<div class="deck">
{''.join(html_slides)}
</div>
<div class="hint">Keys: ← → · Space · Home/End · F fullscreen</div>
<script>
let i=0; const slides=[...document.querySelectorAll('.slide')];
function show(n){{ i=(n+slides.length)%slides.length; slides.forEach((s,k)=>s.classList.toggle('active',k===i)); location.hash='s'+i; }}
document.addEventListener('keydown', e=>{{
  if(['ArrowRight','PageDown',' ','Enter'].includes(e.key)) show(i+1);
  if(['ArrowLeft','PageUp','Backspace'].includes(e.key)) show(i-1);
  if(e.key==='Home') show(0);
  if(e.key==='End') show(slides.length-1);
  if(e.key==='f'||e.key==='F') document.documentElement.requestFullscreen?.();
}});
document.body.addEventListener('click', e=>{{ if(e.target.closest('a,button')) return; show(i+(e.clientX>innerWidth/2?1:-1)); }});
const m=location.hash.match(/s(\\d+)/); show(m?+m[1]:0);
</script>
</body>
</html>
"""

OUT.write_text(html, encoding="utf-8")
print(f"Wrote {OUT} ({len(slides)} slides)")
