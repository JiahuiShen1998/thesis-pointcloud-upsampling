"""Audit active LaTeX citations and local papers without network access."""
from pathlib import Path
import sys, re, json, hashlib

R = Path(__file__).resolve().parents[1]
A = R / 'audit/citation_coverage_20260914'
A.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(R / '.python-deps'))
import pymupdf

def uncomment(t):
    return re.sub(r'(?<!\\)%[^\n]*', '', t)

def walk(p, seen=None):
    seen = set() if seen is None else seen
    if p in seen:
        return seen
    seen.add(p)
    for v in re.findall(r'\\(?:input|include|tableinput)\{([^}]+)\}', uncomment(p.read_text(encoding='utf-8'))):
        q = R / v
        walk(q if q.suffix else q.with_suffix('.tex'), seen)
    return seen

def cite_list(t):
    return [k.strip() for m in re.findall(r'\\cite\w*(?:\[[^]]*\])*\{([^}]+)\}', uncomment(t)) for k in m.split(',')]

def parse_bib(t):
    out = {}
    for m in re.finditer(r'@(\w+)\s*\{([^,]+),', t):
        fields = {}
        i = m.end()
        while i < len(t):
            f = re.match(r'\s*,?\s*(\w+)\s*=\s*', t[i:])
            if not f:
                break
            i += f.end()
            if t[i] != '{':
                raise ValueError('Expected brace-delimited field')
            start = i + 1
            depth = 1
            i += 1
            while depth:
                if t[i] == '{' and t[i-1] != '\\': depth += 1
                if t[i] == '}' and t[i-1] != '\\': depth -= 1
                i += 1
            fields[f.group(1).lower()] = t[start:i-1]
        out[m.group(2).strip()] = dict(type=m.group(1).lower(), **fields)
    return out

entries = parse_bib((R/'bibfiles/references.bib').read_text(encoding='utf-8'))
active = walk(R/'thesis.tex')
occurrences = []
for p in sorted(active):
    for n, line in enumerate(p.read_text(encoding='utf-8').splitlines(), 1):
        for key in cite_list(line):
            occurrences.append(dict(key=key, file=p.relative_to(R).as_posix(), line=n))
keys = sorted({o['key'] for o in occurrences})
bblkeys = re.findall(r'\\bibitem(?:\[[^]]*\])?\{([^}]+)\}', (R/'build/current/thesis.bbl').read_text(encoding='utf-8'))
chapters = []
for p in sorted((R/'texfiles').glob('0[1-6]_*.tex')):
    if not re.search(r'\\chapter\{', p.read_text(encoding='utf-8')):
        continue
    chapterkeys = [k for q in walk(p) for k in cite_list(q.read_text(encoding='utf-8'))]
    chapters.append(dict(file=p.name, citation_mentions=len(chapterkeys), unique_sources=len(set(chapterkeys)), paper_sources=sorted(k for k in set(chapterkeys) if entries[k]['type'] in ('article','inproceedings')), record_sources=sorted(k for k in set(chapterkeys) if entries[k]['type']=='unpublished')))
log = (R/'build/current/thesis.log').read_text(encoding='utf-8', errors='replace')
report = dict(active_files=len(active), cited_sources=len(keys), paper_sources=sum(entries[k]['type'] in ('article','inproceedings') for k in keys), unpublished_records=sum(entries[k]['type']=='unpublished' for k in keys), software_or_web_sources=sum(entries[k]['type']=='misc' for k in keys), chapters=chapters, missing_bib_keys=sorted(set(keys)-entries.keys()), missing_in_bbl=sorted(set(keys)-set(bblkeys)), uncited_in_bbl=sorted(set(bblkeys)-set(keys)), undefined_warnings=[l for l in log.splitlines() if re.search(r'undefined|Citation.*Warning', l, re.I)], canonical_matches_build=hashlib.sha256((R/'thesis.pdf').read_bytes()).hexdigest()==hashlib.sha256((R/'build/current/thesis.pdf').read_bytes()).hexdigest(), occurrences=occurrences)
(A/'coverage.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print('COVERAGE', json.dumps({k:v for k,v in report.items() if k!='occurrences'}), flush=True)

paper_info = []
textdir = A/'paper_text'
textdir.mkdir(exist_ok=True)
for k in keys:
    if entries[k]['type'] not in ('article','inproceedings'):
        continue
    f = R/'bibfiles/papers'/f'{k}.pdf'
    if not f.exists():
        paper_info.append(dict(key=k, missing=True))
        continue
    doc = pymupdf.open(f)
    pages = [p.get_text() for p in doc]
    joined = '\n'.join(f'=== PDF PAGE {i+1} ===\n{t}' for i,t in enumerate(pages))
    (textdir/f'{k}.txt').write_text(joined, encoding='utf-8')
    paper_info.append(dict(key=k, pages=len(doc), sha256=hashlib.sha256(f.read_bytes()).hexdigest(), first_page=pages[0][:2400], doi_strings=sorted(set(re.findall(r'10\.\d{4,9}/[^\s<>]+', pages[0])))))
(A/'local_papers.json').write_text(json.dumps(paper_info, ensure_ascii=False, indent=2), encoding='utf-8')
print('LOCAL_PAPERS', len(paper_info), flush=True)
