#!/usr/bin/env python3
"""Build the complete English dossier from its reviewed Markdown source.

Run from any directory with Python and reportlab installed. The PDF is written
beside the Markdown in presentation/modelnet40_materials. Numeric tables have
one authoritative source, avoiding drift between the Markdown and PDF editions.
"""
from __future__ import annotations
import re
from html import escape
from pathlib import Path
import reportlab
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'presentation/modelnet40_materials').is_dir())
SOURCE = ROOT / 'presentation/modelnet40_materials/ModelNet40_Thesis_Complete_Dossier.md'
OUT = SOURCE.with_suffix('.pdf')
FONTS = [Path('C:/Windows/Fonts/arial.ttf'), Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'), Path(reportlab.__file__).parent / 'fonts/Vera.ttf']
FONT = next(p for p in FONTS if p.exists())
pdfmetrics.registerFont(TTFont('ArchiveSans', str(FONT)))
pdfmetrics.registerFontFamily('ArchiveSans', normal='ArchiveSans', bold='ArchiveSans', italic='ArchiveSans', boldItalic='ArchiveSans')
WIDTH = A4[0] - 3.6 * cm
BODY = ParagraphStyle('body', fontName='ArchiveSans', fontSize=9, leading=13, spaceAfter=6, splitLongWords=True)
CELL = ParagraphStyle('cell', parent=BODY, fontSize=7, leading=9, spaceAfter=0)
HEADINGS = {n: ParagraphStyle(f'h{n}', parent=BODY, fontSize={1:19, 2:14, 3:11}[n], leading={1:24, 2:19, 3:15}[n], spaceBefore=12, spaceAfter=8, keepWithNext=True, textColor=colors.HexColor('#17324d')) for n in (1, 2, 3)}

def inline(text):
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = escape(text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    return text

def paragraph(text, style=BODY):
    return Paragraph(inline(text), style)

def table(lines):
    rows = [[c.strip() for c in line.strip().strip('|').split('|')] for line in lines]
    rows = [row for row in rows if not all(re.fullmatch(r':?-+:?', c) for c in row)]
    cols = len(rows[0])
    assert all(len(row) == cols for row in rows), 'Inconsistent table column count'
    # Allocate more space to descriptive columns while keeping every table on page.
    weights = [max(7, min(50, max(len(row[i]) for row in rows) ** 0.72)) for i in range(cols)]
    widths = [WIDTH * w / sum(weights) for w in weights]
    result = Table([[paragraph(c, CELL) for c in row] for row in rows], colWidths=widths, repeatRows=1, hAlign='LEFT')
    result.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dce7f1')),('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),0.3,colors.HexColor('#bbc6ce')),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    return result

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('ArchiveSans', 8)
    canvas.setFillColor(colors.HexColor('#52606d'))
    canvas.drawString(1.8*cm, 1.0*cm, 'ModelNet40 Experimental Dossier')
    canvas.drawRightString(A4[0]-1.8*cm, 1.0*cm, str(doc.page))
    canvas.restoreState()

def main():
    lines = SOURCE.read_text(encoding='utf-8').splitlines()
    story = []; i = 0
    while i < len(lines):
        line = lines[i].strip(); i += 1
        if not line or line == '---':
            continue
        if line.startswith('|'):
            block = [line]
            while i < len(lines) and lines[i].strip().startswith('|'):
                block.append(lines[i].strip()); i += 1
            story.extend([table(block), Spacer(1,8)])
        elif line.startswith('#'):
            level = min(3, len(line)-len(line.lstrip('#')))
            story.append(paragraph(line.lstrip('#').strip(), HEADINGS[level]))
        else:
            line = line.removeprefix('> ').removeprefix('- ')
            story.append(paragraph(line))
    SimpleDocTemplate(str(OUT),pagesize=A4,leftMargin=1.8*cm,rightMargin=1.8*cm,topMargin=1.5*cm,bottomMargin=1.6*cm,title='ModelNet40 Point-Cloud Upsampling: Complete Experimental Dossier',author='Jiahui Shen').build(story,onFirstPage=footer,onLaterPages=footer)
    print(OUT)

if __name__ == '__main__':
    main()
