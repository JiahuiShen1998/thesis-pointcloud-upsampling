"""Local PDF checks for the citation revision; no network access."""
from pathlib import Path
import sys, re, json, hashlib, runpy
R = Path(__file__).resolve().parents[1]
A = R / 'audit/citation_coverage_20260914'
runpy.run_path(str(R/'scripts/audit_citation_sources.py'))
sys.path.insert(0,str(R/'.python-deps'))
import pymupdf
doc = pymupdf.open(R/'build/current/thesis.pdf')
texts = [p.get_text() for p in doc]
alltext = '\n'.join(texts)
markers = ['Fan et al. define','downstream architecture mediates','MV3D evaluates KITTI','Bibliography']
locations = {m:[i+1 for i,t in enumerate(texts) if m in t] for m in markers}
render = {i-1 for m, pages in locations.items() for i in pages[-1:]}
render.update(range(max(0,len(doc)-5),len(doc)))
for i in sorted(render):
    doc[i].get_pixmap(matrix=pymupdf.Matrix(1.2,1.2), alpha=False).save(str(A/f'page_{i+1}.png'))
bbl = (R/'build/current/thesis.bbl').read_text(encoding='utf-8')
pdf_report = dict(pages=len(doc), citations=len(re.findall(r'\\bibitem\{',bbl)), rendered_urls=bbl.count('\\url{'), marker_pages=locations, appendix='Appendix' in alltext, tulip='TULIP' in alltext, spupmd='SPU-PMD' in alltext, sha256=hashlib.sha256((R/'build/current/thesis.pdf').read_bytes()).hexdigest())
(A/'pdf_check.json').write_text(json.dumps(pdf_report,indent=2),encoding='utf-8')
print('PDF',json.dumps(pdf_report),flush=True)
