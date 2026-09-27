#!/usr/bin/env python3
"""Build the combined PointRCNN generated-ratio result table and analysis."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO = Path(__file__).resolve().parents[1]
FINE_ROOT = REPO / "results/kitti_x4_detector_recovery_fine_ratio_v2_20260725"
COARSE_ROOT = REPO / "results/kitti_x4_detector_recovery_ratio_v1_20260723"
OUT = REPO / "results/kitti_x4_detector_recovery_combined_analysis_20260725"
FINE_CSV = FINE_ROOT / "reports/fine_ratio_screen_ap_summary.csv"
COARSE_CSV = COARSE_ROOT / "reports/screen_ap_summary.csv"

METHODS = ("pdans", "pu_gcn", "pu_edgeformer", "pu_net")
METHOD_LABELS = {
    "pdans": "PDANS",
    "pu_gcn": "PU-GCN",
    "pu_edgeformer": "PU-EdgeFormer",
    "pu_net": "PU-Net",
}
LINES = ("original", "downsampled")
LINE_LABELS = {"original": "Original", "downsampled": "Downsampled"}
COLORS = {
    "pdans": "#1b9e77",
    "pu_gcn": "#377eb8",
    "pu_edgeformer": "#ff7f00",
    "pu_net": "#e41a1c",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def line_method(variant: str) -> tuple[str, str]:
    if variant.startswith("original_x4_"):
        return "original", variant.removeprefix("original_x4_")
    if variant.startswith("downsampled_x4_"):
        return "downsampled", variant.removeprefix("downsampled_x4_")
    raise ValueError(f"unrecognized method variant: {variant}")


def fmt(value: float) -> str:
    return f"{value:.2f}"


def signed(value: float) -> str:
    return f"{value:+.2f}"


def complete_ap_metrics(summary_path: str) -> dict[str, float]:
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    metrics = summary["ap_r40_percent"]
    output: dict[str, float] = {}
    for source, label in (
        ("bbox_ap", "bbox"),
        ("bev_ap", "bev"),
        ("3d_ap", "3d"),
        ("aos_ap", "aos"),
    ):
        for difficulty in ("easy", "moderate", "hard"):
            output[f"ap_{label}_car_{difficulty}_r40"] = metrics[source][difficulty]
    return output


def excel_column(index: int) -> str:
    value = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        value = chr(65 + remainder) + value
    return value


def write_single_sheet_xlsx(
    path: Path,
    columns: list[str],
    rows: list[dict[str, object]],
    cell_styles: dict[tuple[int, str], int] | None = None,
) -> None:
    cell_styles = cell_styles or {}
    sheet_rows: list[str] = []
    header_cells = []
    for index, column in enumerate(columns, start=1):
        coordinate = f"{excel_column(index)}1"
        header_cells.append(
            f'<c r="{coordinate}" t="inlineStr" s="1"><is><t>{escape(column)}</t></is></c>'
        )
    sheet_rows.append(f'<row r="1" ht="30" customHeight="1">{"".join(header_cells)}</row>')

    for row_index, row in enumerate(rows, start=2):
        cells: list[str] = []
        for column_index, column in enumerate(columns, start=1):
            coordinate = f"{excel_column(column_index)}{row_index}"
            value = row.get(column, "")
            if value == "" or value is None:
                continue
            requested_style = cell_styles.get((row_index - 2, column))
            if isinstance(value, (int, float)):
                style = requested_style or 2
                cells.append(f'<c r="{coordinate}" s="{style}"><v>{value}</v></c>')
            else:
                style_attribute = f' s="{requested_style}"' if requested_style else ""
                cells.append(
                    f'<c r="{coordinate}" t="inlineStr"{style_attribute}><is><t>'
                    f"{escape(str(value))}</t></is></c>"
                )
        sheet_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')

    column_xml: list[str] = []
    for index, column in enumerate(columns, start=1):
        if column == "Source":
            width = 70
        elif column.startswith("Best"):
            width = 55
        elif column in ("Protocol", "RowKind"):
            width = 28
        elif column in ("Method", "Policy"):
            width = 22
        elif column in ("Line", "Status"):
            width = 14
        else:
            width = 19
        column_xml.append(
            f'<col min="{index}" max="{index}" width="{width}" customWidth="1"/>'
        )
    last_column = excel_column(len(columns))
    last_row = len(rows) + 1
    worksheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetViews><sheetView workbookViewId="0">'
        '<pane xSplit="6" ySplit="1" topLeftCell="G2" activePane="bottomRight" state="frozen"/>'
        '</sheetView></sheetViews>'
        '<sheetFormatPr defaultRowHeight="18"/>'
        f'<cols>{"".join(column_xml)}</cols>'
        f'<sheetData>{"".join(sheet_rows)}</sheetData>'
        f'<autoFilter ref="A1:{last_column}{last_row}"/>'
        '</worksheet>'
    )
    styles = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <numFmts count="1"><numFmt numFmtId="164" formatCode="0.00"/></numFmts>
  <fonts count="3">
    <font><sz val="11"/><name val="Calibri"/></font>
    <font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/></font>
    <font><b/><sz val="11"/><name val="Calibri"/></font>
  </fonts>
  <fills count="5">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFC6EFCE"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFFFD966"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="5">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf>
    <xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>
    <xf numFmtId="164" fontId="2" fillId="3" borderId="0" xfId="0" applyNumberFormat="1" applyFont="1" applyFill="1"/>
    <xf numFmtId="164" fontId="2" fillId="4" borderId="0" xfId="0" applyNumberFormat="1" applyFont="1" applyFill="1"/>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>"""
    workbook = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="All Comparison" sheetId="1" r:id="rId1"/></sheets>
</workbook>"""
    workbook_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""
    package_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>"""
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", package_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)
        archive.writestr("xl/styles.xml", styles)


def write_multi_sheet_xlsx(
    path: Path,
    sheets: list[
        tuple[
            str,
            list[str],
            list[dict[str, object]],
            dict[tuple[int, str], int],
        ]
    ],
) -> None:
    worksheet_xml: list[str] = []
    styles = ""
    with TemporaryDirectory(prefix="pointrcnn-comparison-") as temp_dir:
        temp_root = Path(temp_dir)
        for index, (_, columns, rows, cell_styles) in enumerate(sheets, start=1):
            temporary = temp_root / f"sheet{index}.xlsx"
            write_single_sheet_xlsx(temporary, columns, rows, cell_styles)
            with ZipFile(temporary) as archive:
                worksheet_xml.append(
                    archive.read("xl/worksheets/sheet1.xml").decode("utf-8")
                )
                if not styles:
                    styles = archive.read("xl/styles.xml").decode("utf-8")

    workbook_sheets = "".join(
        f'<sheet name="{escape(name)}" sheetId="{index}" r:id="rId{index}"/>'
        for index, (name, _, _, _) in enumerate(sheets, start=1)
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{workbook_sheets}</sheets></workbook>"
    )
    relationships = "".join(
        f'<Relationship Id="rId{index}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        f'Target="worksheets/sheet{index}.xml"/>'
        for index in range(1, len(sheets) + 1)
    )
    relationships += (
        f'<Relationship Id="rId{len(sheets) + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{relationships}</Relationships>"
    )
    sheet_overrides = "".join(
        f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for index in range(1, len(sheets) + 1)
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        f"{sheet_overrides}"
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        "</Types>"
    )
    package_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", package_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        for index, xml in enumerate(worksheet_xml, start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", xml)
        archive.writestr("xl/styles.xml", styles)


def main() -> int:
    fine_rows = read_csv(FINE_CSV)
    coarse_rows = read_csv(COARSE_CSV)
    if any(row["status"] != "PASS" for row in fine_rows + coarse_rows):
        raise RuntimeError("combined analysis requires every input result row to be PASS")

    baseline: dict[str, float] = {}
    fine: dict[tuple[str, str], dict[float, float]] = defaultdict(dict)
    controls: dict[str, dict[float, float]] = defaultdict(dict)
    coarse: dict[tuple[str, str], dict[float, float]] = defaultdict(dict)

    for row in fine_rows:
        ratio = float(row["generated_ratio"]) * 100.0
        ap = float(row["3d_ap_moderate"])
        if row["row_kind"] == "baseline_reference":
            line = "original" if row["variant"] == "original_baseline" else "downsampled"
            baseline[line] = ap
        elif row["row_kind"] == "generated_ratio":
            line, method = line_method(row["variant"])
            fine[(line, method)][ratio] = ap
        elif row["row_kind"] == "observed_fill_control":
            line = "original" if row["variant"].startswith("original_") else "downsampled"
            controls[line][ratio] = ap

    for row in coarse_rows:
        line, method = line_method(row["variant"])
        coarse[(line, method)][float(row["generated_ratio"]) * 100.0] = float(
            row["3d_ap_moderate"]
        )

    if set(baseline) != set(LINES):
        raise RuntimeError(f"missing baselines: found {sorted(baseline)}")

    OUT.mkdir(parents=True, exist_ok=True)
    combined_rows: list[dict[str, object]] = []
    for line in LINES:
        combined_rows.append(
            {
                "line": line,
                "method": "baseline",
                "protocol": "baseline",
                "ratio_percent": 0.0,
                "ap_3d_car_moderate_r40": baseline[line],
                "delta_vs_baseline": 0.0,
                "observed_fill_control_ap": "",
                "delta_vs_matched_control": "",
                "frames": 256,
                "status": "PASS",
            }
        )
        for method in METHODS:
            for ratio, ap in sorted(fine[(line, method)].items()):
                control = controls[line][ratio]
                combined_rows.append(
                    {
                        "line": line,
                        "method": method,
                        "protocol": "fine_nested_e2_replacement",
                        "ratio_percent": ratio,
                        "ap_3d_car_moderate_r40": ap,
                        "delta_vs_baseline": ap - baseline[line],
                        "observed_fill_control_ap": control,
                        "delta_vs_matched_control": ap - control,
                        "frames": 256,
                        "status": "PASS",
                    }
                )
            for ratio, ap in sorted(coarse[(line, method)].items()):
                combined_rows.append(
                    {
                        "line": line,
                        "method": method,
                        "protocol": "coarse_independent_sampling",
                        "ratio_percent": ratio,
                        "ap_3d_car_moderate_r40": ap,
                        "delta_vs_baseline": ap - baseline[line],
                        "observed_fill_control_ap": "",
                        "delta_vs_matched_control": "",
                        "frames": 256,
                        "status": "PASS",
                    }
                )

    combined_csv = OUT / "combined_ratio_results.csv"
    with combined_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined_rows[0]))
        writer.writeheader()
        writer.writerows(combined_rows)

    all_metric_rows: list[dict[str, object]] = []
    for row in fine_rows:
        if row["row_kind"] == "baseline_reference":
            line = "original" if row["variant"] == "original_baseline" else "downsampled"
            method = "baseline"
            protocol = "baseline"
        elif row["row_kind"] == "observed_fill_control":
            line = "original" if row["variant"].startswith("original_") else "downsampled"
            method = "observed_fill_control"
            protocol = "fine_nested_e2_replacement"
        else:
            line, method = line_method(row["variant"])
            protocol = "fine_nested_e2_replacement"
        all_metric_rows.append(
            {
                "line": line,
                "method": method,
                "row_kind": row["row_kind"],
                "protocol": protocol,
                "policy": row["policy"],
                "ratio_percent": float(row["generated_ratio"]) * 100.0,
                **complete_ap_metrics(row["summary_path"]),
                "frames": row["prediction_files"],
                "status": row["status"],
                "summary_path": row["summary_path"],
            }
        )
    for row in coarse_rows:
        line, method = line_method(row["variant"])
        all_metric_rows.append(
            {
                "line": line,
                "method": method,
                "row_kind": "generated_ratio",
                "protocol": "coarse_independent_sampling",
                "policy": row["policy"],
                "ratio_percent": float(row["generated_ratio"]) * 100.0,
                **complete_ap_metrics(row["summary_path"]),
                "frames": row["prediction_files"],
                "status": row["status"],
                "summary_path": row["summary_path"],
            }
        )
    all_metrics_csv = OUT / "combined_all_difficulty_metrics.csv"
    with all_metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_metric_rows[0]))
        writer.writeheader()
        writer.writerows(all_metric_rows)

    metric_columns = [
        f"ap_{metric}_car_{difficulty}_r40"
        for metric in ("bbox", "bev", "3d", "aos")
        for difficulty in ("easy", "moderate", "hard")
    ]
    metric_short_name = {
        f"ap_{metric}_car_{difficulty}_r40": (
            f"{metric.upper()}-{difficulty[0].upper()}"
        )
        for metric in ("bbox", "bev", "3d", "aos")
        for difficulty in ("easy", "moderate", "hard")
    }
    generated_metric_rows = [
        row for row in all_metric_rows if row["row_kind"] == "generated_ratio"
    ]
    best_same_ratio: dict[tuple[str, str, float, str], float] = {}
    best_in_protocol: dict[tuple[str, str, str], float] = {}
    for row in generated_metric_rows:
        line = str(row["line"])
        protocol = str(row["protocol"])
        ratio = float(row["ratio_percent"])
        for column in metric_columns:
            value = float(row[column])
            same_key = (line, protocol, ratio, column)
            protocol_key = (line, protocol, column)
            best_same_ratio[same_key] = max(best_same_ratio.get(same_key, value), value)
            best_in_protocol[protocol_key] = max(
                best_in_protocol.get(protocol_key, value), value
            )
    baseline_metric_rows = {
        str(row["line"]): row
        for row in all_metric_rows
        if row["row_kind"] == "baseline_reference"
    }
    control_metric_rows = {
        (str(row["line"]), float(row["ratio_percent"])): row
        for row in all_metric_rows
        if row["row_kind"] == "observed_fill_control"
    }
    comparison_rows: list[dict[str, object]] = []
    identifier_columns = [
        "line",
        "method",
        "row_kind",
        "protocol",
        "policy",
        "ratio_percent",
        "frames",
        "status",
    ]
    for row in all_metric_rows:
        line = str(row["line"])
        ratio = float(row["ratio_percent"])
        line_baseline_row = baseline_metric_rows[line]
        original_baseline_row = baseline_metric_rows["original"]
        matched_control = (
            control_metric_rows.get((line, ratio))
            if row["row_kind"] == "generated_ratio"
            and row["protocol"] == "fine_nested_e2_replacement"
            else None
        )
        output_row = {column: row[column] for column in identifier_columns}
        for column in metric_columns:
            value = float(row[column])
            output_row[column] = value
            output_row[f"original_baseline__{column}"] = float(
                original_baseline_row[column]
            )
            output_row[f"delta_vs_original_baseline__{column}"] = value - float(
                original_baseline_row[column]
            )
            output_row[f"line_baseline__{column}"] = float(
                line_baseline_row[column]
            )
            output_row[f"delta_vs_line_baseline__{column}"] = value - float(
                line_baseline_row[column]
            )
            output_row[f"matched_control__{column}"] = (
                float(matched_control[column]) if matched_control else ""
            )
            output_row[f"delta_vs_control__{column}"] = (
                value - float(matched_control[column]) if matched_control else ""
            )
        if row["row_kind"] == "generated_ratio":
            same_wins = [
                metric_short_name[column]
                for column in metric_columns
                if abs(
                    float(row[column])
                    - best_same_ratio[
                        (
                            line,
                            str(row["protocol"]),
                            ratio,
                            column,
                        )
                    ]
                )
                < 1e-9
            ]
            protocol_wins = [
                metric_short_name[column]
                for column in metric_columns
                if abs(
                    float(row[column])
                    - best_in_protocol[(line, str(row["protocol"]), column)]
                )
                < 1e-9
            ]
        else:
            same_wins = []
            protocol_wins = []
        output_row["best_at_same_ratio_metrics"] = ", ".join(same_wins)
        output_row["best_across_ratios_in_protocol_metrics"] = ", ".join(
            protocol_wins
        )
        output_row["summary_path"] = row["summary_path"]
        comparison_rows.append(output_row)
    comparison_columns = identifier_columns + [
        "best_at_same_ratio_metrics",
        "best_across_ratios_in_protocol_metrics",
        "summary_path",
    ]
    for column in metric_columns:
        comparison_columns.extend(
            [
                column,
                f"original_baseline__{column}",
                f"delta_vs_original_baseline__{column}",
                f"line_baseline__{column}",
                f"delta_vs_line_baseline__{column}",
                f"matched_control__{column}",
                f"delta_vs_control__{column}",
            ]
        )
    comparison_csv = OUT / "all_methods_all_ratios_all_metrics_comparison.csv"
    with comparison_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_columns)
        writer.writeheader()
        writer.writerows(comparison_rows)

    difficulty_labels = {"easy": "E", "moderate": "M", "hard": "H"}
    excel_identifier_map = {
        "line": "Line",
        "method": "Method",
        "row_kind": "RowKind",
        "protocol": "Protocol",
        "policy": "Policy",
        "ratio_percent": "GeneratedRatioPct",
        "frames": "Frames",
        "status": "Status",
        "best_at_same_ratio_metrics": "BestAtSameRatioMetrics",
        "best_across_ratios_in_protocol_metrics": "BestAcrossProtocolMetrics",
        "summary_path": "Source",
    }
    excel_columns = list(excel_identifier_map.values())
    excel_metric_map: list[
        tuple[str, str, str, str, str, str, str, str]
    ] = []
    for metric in ("bbox", "bev", "3d", "aos"):
        for difficulty in ("easy", "moderate", "hard"):
            source = f"ap_{metric}_car_{difficulty}_r40"
            prefix = f"{metric.upper()}_{difficulty_labels[difficulty]}"
            excel_metric_map.append(
                (
                    source,
                    f"{prefix}_AP",
                    f"{prefix}_OriginalBaselineAP",
                    f"{prefix}_DeltaOriginalBaseline",
                    f"{prefix}_LineBaselineAP",
                    f"{prefix}_DeltaLineBaseline",
                    f"{prefix}_ControlAP",
                    f"{prefix}_DeltaControl",
                )
            )
            excel_columns.extend(excel_metric_map[-1][1:])
    excel_rows: list[dict[str, object]] = []
    excel_styles: dict[tuple[int, str], int] = {}
    for row_index, row in enumerate(comparison_rows):
        excel_row = {
            target: row[source] for source, target in excel_identifier_map.items()
        }
        for (
            source,
            ap_name,
            original_baseline_name,
            original_delta_name,
            line_baseline_name,
            line_delta_name,
            control_name,
            control_delta_name,
        ) in excel_metric_map:
            excel_row[ap_name] = row[source]
            excel_row[original_baseline_name] = row[f"original_baseline__{source}"]
            excel_row[original_delta_name] = row[
                f"delta_vs_original_baseline__{source}"
            ]
            excel_row[line_baseline_name] = row[f"line_baseline__{source}"]
            excel_row[line_delta_name] = row[f"delta_vs_line_baseline__{source}"]
            excel_row[control_name] = row[f"matched_control__{source}"]
            excel_row[control_delta_name] = row[f"delta_vs_control__{source}"]
            if row["row_kind"] == "generated_ratio":
                same_key = (
                    str(row["line"]),
                    str(row["protocol"]),
                    float(row["ratio_percent"]),
                    source,
                )
                protocol_key = (str(row["line"]), str(row["protocol"]), source)
                value = float(row[source])
                if abs(value - best_in_protocol[protocol_key]) < 1e-9:
                    excel_styles[(row_index, ap_name)] = 4
                elif abs(value - best_same_ratio[same_key]) < 1e-9:
                    excel_styles[(row_index, ap_name)] = 3
        excel_rows.append(excel_row)
    original_indices = [
        index
        for index, row in enumerate(comparison_rows)
        if row["line"] == "original"
    ]
    original_baseline_index = next(
        index
        for index, row in enumerate(comparison_rows)
        if row["line"] == "original" and row["row_kind"] == "baseline_reference"
    )
    downsampled_indices = [original_baseline_index] + [
        index
        for index, row in enumerate(comparison_rows)
        if row["line"] == "downsampled"
    ]
    for filename, indices in (
        ("original_line_vs_original_baseline.csv", original_indices),
        ("downsampled_line_vs_original_baseline.csv", downsampled_indices),
    ):
        with (OUT / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=comparison_columns)
            writer.writeheader()
            writer.writerows(comparison_rows[index] for index in indices)

    def excel_subset(
        indices: list[int],
    ) -> tuple[list[dict[str, object]], dict[tuple[int, str], int]]:
        rows = [excel_rows[index] for index in indices]
        styles = {
            (local_index, column): excel_styles[(global_index, column)]
            for local_index, global_index in enumerate(indices)
            for column in excel_columns
            if (global_index, column) in excel_styles
        }
        return rows, styles

    original_excel_rows, original_excel_styles = excel_subset(original_indices)
    downsampled_excel_rows, downsampled_excel_styles = excel_subset(
        downsampled_indices
    )
    write_multi_sheet_xlsx(
        OUT / "all_methods_all_ratios_all_metrics_comparison.xlsx",
        [
            (
                "Original Line",
                excel_columns,
                original_excel_rows,
                original_excel_styles,
            ),
            (
                "Downsampled Line",
                excel_columns,
                downsampled_excel_rows,
                downsampled_excel_styles,
            ),
        ],
    )

    def marked_ap(row: dict[str, object], column: str) -> str:
        text = f"{float(row[column]):.2f}"
        if row["row_kind"] != "generated_ratio":
            return text
        key = (
            str(row["line"]),
            str(row["protocol"]),
            float(row["ratio_percent"]),
            column,
        )
        if abs(float(row[column]) - best_same_ratio[key]) < 1e-9:
            return f"**{text}★**"
        return text

    bev_3d_report = [
        "# PointRCNN完整BEV/3D AP R40对比",
        "",
        "所有数值均来自相同256帧Car筛选集；E/M/H分别表示Easy/Moderate/Hard。",
        "同一实验线、协议和比例下，四种方法的最优值使用粗体和★标注。",
        "",
    ]
    sections = (
        ("Baseline", "baseline"),
        ("细比例方法", "fine_nested_e2_replacement"),
        ("粗比例方法", "coarse_independent_sampling"),
    )
    for title, protocol in sections:
        bev_3d_report.extend([f"## {title}", ""])
        for line in LINES:
            rows = [
                row
                for row in all_metric_rows
                if row["line"] == line
                and row["protocol"] == protocol
                and (
                    protocol == "baseline"
                    or row["row_kind"] == "generated_ratio"
                )
            ]
            if not rows:
                continue
            bev_3d_report.extend(
                [
                    f"### {LINE_LABELS[line]}",
                    "",
                    "| 方法 | 比例 | BEV-E | BEV-M | BEV-H | 3D-E | 3D-M | 3D-H |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in rows:
                label = (
                    "Baseline"
                    if row["method"] == "baseline"
                    else METHOD_LABELS[str(row["method"])]
                )
                bev_3d_report.append(
                    f"| {label} | {float(row['ratio_percent']):g}% | "
                    f"{marked_ap(row, 'ap_bev_car_easy_r40')} | "
                    f"{marked_ap(row, 'ap_bev_car_moderate_r40')} | "
                    f"{marked_ap(row, 'ap_bev_car_hard_r40')} | "
                    f"{marked_ap(row, 'ap_3d_car_easy_r40')} | "
                    f"{marked_ap(row, 'ap_3d_car_moderate_r40')} | "
                    f"{marked_ap(row, 'ap_3d_car_hard_r40')} |"
                )
            bev_3d_report.append("")
    control_rows = [
        row for row in all_metric_rows if row["row_kind"] == "observed_fill_control"
    ]
    bev_3d_report.extend(["## 真实点匹配对照", ""])
    for line in LINES:
        bev_3d_report.extend(
            [
                f"### {LINE_LABELS[line]}",
                "",
                "| 比例 | BEV-E | BEV-M | BEV-H | 3D-E | 3D-M | 3D-H |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in control_rows:
            if row["line"] != line:
                continue
            bev_3d_report.append(
                f"| {float(row['ratio_percent']):g}% | "
                f"{float(row['ap_bev_car_easy_r40']):.2f} | "
                f"{float(row['ap_bev_car_moderate_r40']):.2f} | "
                f"{float(row['ap_bev_car_hard_r40']):.2f} | "
                f"{float(row['ap_3d_car_easy_r40']):.2f} | "
                f"{float(row['ap_3d_car_moderate_r40']):.2f} | "
                f"{float(row['ap_3d_car_hard_r40']):.2f} |"
            )
        bev_3d_report.append("")
    (OUT / "complete_bev_3d_comparison.md").write_text(
        "\n".join(bev_3d_report), encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    for column, line in enumerate(LINES):
        ax = axes[0, column]
        for method in METHODS:
            points = fine[(line, method)]
            xs = [0.0] + sorted(points)
            ys = [baseline[line]] + [points[x] for x in sorted(points)]
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2,
                color=COLORS[method],
                label=METHOD_LABELS[method],
            )
        control_x = [0.0] + sorted(controls[line])
        control_y = [baseline[line]] + [controls[line][x] for x in sorted(controls[line])]
        ax.plot(
            control_x,
            control_y,
            marker="s",
            linestyle="--",
            linewidth=1.8,
            color="#555555",
            label="Observed-fill control",
        )
        ax.axhline(baseline[line], color="#222222", linestyle=":", linewidth=1.2)
        ax.set_title(f"{LINE_LABELS[line]} — nested fine ratios")
        ax.set_xlabel("Generated points in detector input (%)")
        ax.set_ylabel("Car 3D AP R40 Moderate (%)")
        ax.set_xticks([0, 2.5, 5, 7.5, 10])
        ax.grid(alpha=0.25)

        ax = axes[1, column]
        for method in METHODS:
            points = coarse[(line, method)]
            xs = [0.0] + sorted(points)
            ys = [baseline[line]] + [points[x] for x in sorted(points)]
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2,
                color=COLORS[method],
                label=METHOD_LABELS[method],
            )
        ax.axhline(baseline[line], color="#222222", linestyle=":", linewidth=1.2)
        ax.set_title(f"{LINE_LABELS[line]} — independent coarse ratios")
        ax.set_xlabel("Generated points in detector input (%)")
        ax.set_ylabel("Car 3D AP R40 Moderate (%)")
        ax.set_xticks([0, 10, 15, 25, 35, 40, 50])
        ax.grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.948),
        ncol=5,
        frameon=False,
    )
    fig.suptitle(
        "PointRCNN sensitivity to generated-point ratio (same 256-frame Car screen)",
        y=0.99,
        fontsize=15,
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.07, top=0.88, hspace=0.28, wspace=0.12)
    figure_png = OUT / "ratio_ap_comparison.png"
    figure_pdf = OUT / "ratio_ap_comparison.pdf"
    fig.savefig(figure_png, dpi=180)
    fig.savefig(figure_pdf)
    plt.close(fig)

    fine_ratios = sorted(next(iter(fine.values())))
    coarse_ratios = sorted(next(iter(coarse.values())))
    report: list[str] = [
        "# PointRCNN上采样生成点比例：完整比较与原因分析",
        "",
        "## 实验完成状态",
        "",
        "- 细比例嵌套实验：4种方法 × 2条线 × 4个比例 = 32/32 PASS。",
        "- 真实观测点匹配对照：2条线 × 4个比例 = 8/8 PASS。",
        "- 粗比例实验：4种方法 × 2条线 × 6个比例 = 48/48 PASS。",
        "- 两个实验使用完全相同的256帧Car筛选集和同一PointRCNN检测器。",
        "- 指标均为Car 3D AP R40 Moderate（%）。",
        "",
        "## 细比例结果：嵌套、方法无关的E2槽位替换",
        "",
    ]

    for line in LINES:
        report.extend(
            [
                f"### {LINE_LABELS[line]}线",
                "",
                "| 方法 | g0 baseline | g2.5 | g5 | g7.5 | g10 | 最佳生成点设置 | Δ baseline |",
                "|---|---:|---:|---:|---:|---:|---|---:|",
            ]
        )
        for method in METHODS:
            points = fine[(line, method)]
            best_ratio, best_ap = max(points.items(), key=lambda item: item[1])
            values = " | ".join(fmt(points[ratio]) for ratio in fine_ratios)
            report.append(
                f"| {METHOD_LABELS[method]} | {fmt(baseline[line])} | {values} | "
                f"g{best_ratio:g} | {signed(best_ap - baseline[line])} |"
            )
        report.append("")

    report.extend(
        [
            "## 同比例真实点替换对照",
            "",
            "| 实验线 | c2.5 | c5 | c7.5 | c10 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for line in LINES:
        values = " | ".join(fmt(controls[line][ratio]) for ratio in fine_ratios)
        report.append(f"| {LINE_LABELS[line]} | {values} |")
    report.extend(
        [
            "",
            "生成点相对匹配真实点对照的差值（方法AP − control AP）：",
            "",
        ]
    )
    for line in LINES:
        report.extend(
            [
                f"### {LINE_LABELS[line]}线",
                "",
                "| 方法 | g2.5 | g5 | g7.5 | g10 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for method in METHODS:
            values = " | ".join(
                signed(fine[(line, method)][ratio] - controls[line][ratio])
                for ratio in fine_ratios
            )
            report.append(f"| {METHOD_LABELS[method]} | {values} |")
        report.append("")

    report.extend(
        [
            "## 粗比例结果：独立比例采样",
            "",
        ]
    )
    for line in LINES:
        report.extend(
            [
                f"### {LINE_LABELS[line]}线",
                "",
                "| 方法 | g0 baseline | g10 | g15 | g25 | g35 | g40 | g50 | g10→g50下降 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method in METHODS:
            points = coarse[(line, method)]
            values = " | ".join(fmt(points[ratio]) for ratio in coarse_ratios)
            drop = points[50.0] - points[10.0]
            report.append(
                f"| {METHOD_LABELS[method]} | {fmt(baseline[line])} | {values} | {signed(drop)} |"
            )
        report.append("")

    report.extend(
        [
            "## 两种协议不能无条件拼成一条精确曲线",
            "",
            "细比例实验从固定的E2 baseline 16,384点母集出发，所有方法替换相同槽位，且比例间嵌套；"
            "粗比例实验对每个方法和比例独立抽取真实点与生成点。两者帧集相同，但采样协议不同。",
            "",
            "| 实验线/方法 | 细协议g10 | 粗协议g10 | 粗−细 |",
            "|---|---:|---:|---:|",
        ]
    )
    for line in LINES:
        for method in METHODS:
            fine_g10 = fine[(line, method)][10.0]
            coarse_g10 = coarse[(line, method)][10.0]
            report.append(
                f"| {LINE_LABELS[line]}/{METHOD_LABELS[method]} | {fmt(fine_g10)} | "
                f"{fmt(coarse_g10)} | {signed(coarse_g10 - fine_g10)} |"
            )

    original_pdans = fine[("original", "pdans")][2.5]
    original_pdans_control = controls["original"][2.5]
    report.extend(
        [
            "",
            "## 核心结论",
            "",
            f"1. **唯一明确超过baseline的候选是Original/PDANS g2.5。** "
            f"其AP为{fmt(original_pdans)}，相对Original baseline "
            f"{fmt(baseline['original'])}提高{signed(original_pdans - baseline['original'])}。",
            f"2. **这{signed(original_pdans - baseline['original'])}不能全部归功于生成点。** "
            f"同槽位使用真实点的c2.5对照已经达到{fmt(original_pdans_control)}；"
            f"PDANS相对匹配对照只高{signed(original_pdans - original_pdans_control)}。"
            "因此主要增益中包含检测器采样/覆盖变化，生成几何的净贡献较小，必须用完整验证集确认。",
            "3. **Original/PU-GCN g2.5仅与baseline持平。** 其相对匹配真实点对照仍明显更低，"
            "不能视为稳定提升。",
            "4. **Downsampled线没有任何方法或比例超过baseline。** 即使只引入2.5%生成点，"
            "四种方法也全部下降；说明当前方法不能恢复下采样丢失的目标证据。",
            "5. **比例越高，退化总体越强。** 从10%增加到50%时，所有方法和两条线均大幅下降；"
            "PDANS最耐受、PU-GCN其次、PU-EdgeFormer更差、当前PU-Net接入最差。",
            "6. **原因不是单一的detector问题。** 真实点对照本身会改变AP，证明PointRCNN固定点数采样"
            "对输入构成敏感；但绝大多数方法又低于同样比例的真实点对照，证明生成点几何/分布质量"
            "也在造成额外损失。",
            "7. **当前PU-Net结果还包含已确认的接入缺陷。** wrapper缺少预训练要求的patch中心化、"
            "尺度归一化和输出逆变换，因此不能把当前曲线当作PU-Net模型能力上限。",
            "",
            "## 建议的下一步",
            "",
            "1. 先在完整KITTI验证集上运行Original/PDANS g2.5及匹配c2.5对照；"
            "只有PDANS显著超过两者，才能确认真正检测增益。",
            "2. 不建议把35%～50%配置送入完整验证集；256帧趋势已经显示其系统性退化。",
            "3. 修复PU-Net归一化/逆变换并使用局部FPS+kNN patch后，重新生成点云再评测。",
            "4. 对所有方法采用保留真实点优先、生成点置信度筛选、目标邻域定向补点，"
            "避免随机生成点替换稀缺真实测量。",
            "5. 若希望较高生成点比例也有效，需要用相同混合比例对PointRCNN进行训练或微调；"
            "当前检测器只适应真实KITTI点分布。",
            "",
            "## 输出文件",
            "",
            "- `combined_ratio_results.csv`：所有协议、比例、方法和相对差值。",
            "- `combined_all_difficulty_metrics.csv`：BBox/BEV/3D/AOS × Easy/Moderate/Hard完整AP。",
            "- `all_methods_all_ratios_all_metrics_comparison.csv`：统一宽表，含全部AP、"
            "相对Original baseline、实验线baseline及细比例匹配control的差值。",
            "- `all_methods_all_ratios_all_metrics_comparison.xlsx`：分Original/Downsampled"
            "两个工作表，二者均以Original baseline为统一参照。",
            "- `original_line_vs_original_baseline.csv`与"
            "`downsampled_line_vs_original_baseline.csv`：两条线的独立宽表。",
            "- `complete_bev_3d_comparison.md`：便于阅读的全部BEV/3D三级难度表。",
            "- `ratio_ap_comparison.png/.pdf`：四面板比例—AP曲线。",
            "- `combined_analysis_report.md`：本报告。",
            "",
        ]
    )

    report_path = OUT / "combined_analysis_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"COMBINED_ANALYSIS_PASS output={OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
