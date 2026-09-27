"""Offline audit of active citations, BibTeX numbering and rendered PDF links."""
from pathlib import Path
import re, json, sys, csv, hashlib, difflib
from collections import Counter

R = Path(__file__).resolve().parents[1]
A = R / 'audit/teacher_email_revision_20260917'
A.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(R / '.python-deps'))
import pymupdf

def uncomment(t):
    return re.sub(r'(?<!\\)%[^\n]*', '', t)

def parse_bib(t):
    out = {}
    for m in re.finditer(r'@(\w+)\s*\{([^,]+),', t):
        fields = {}
        i = m.end()
        while i < len(t):
            f = re.match(r'\s*,?\s*(\w+)\s*=\s*', t[i:])
            if not f: break
            i += f.end()
            assert t[i] == '{'
            start, depth = i + 1, 1
            i += 1
            while depth:
                if t[i] == '{' and t[i-1] != '\\': depth += 1
                if t[i] == '}' and t[i-1] != '\\': depth -= 1
                i += 1
            fields[f.group(1).lower()] = t[start:i-1]
        key = m.group(2).strip()
        assert key not in out, key
        out[key] = dict(type=m.group(1).lower(), **fields)
    return out

entries = parse_bib((R / 'bibfiles/references.bib').read_text(encoding='utf-8'))
events, active = [], []
token = re.compile(r'\\(?:(input|include|tableinput)\{([^}]+)\}|(cite\w*)(?:\[[^]]*\])*\{([^}]+)\})')

def visit(p, chapter=0):
    assert p not in active, f'Duplicate active input: {p}'
    active.append(p)
    t = uncomment(p.read_text(encoding='utf-8'))
    m = re.match(r'0([1-6])_', p.name)
    if m: chapter = int(m.group(1))
    for match in token.finditer(t):
        if match.group(1):
            child = R / match.group(2)
            visit(child if child.suffix else child.with_suffix('.tex'), chapter)
        else:
            for k in match.group(4).split(','):
                events.append(dict(key=k.strip(), chapter=chapter,
                    file=p.relative_to(R).as_posix(), line=t.count('\n', 0, match.start()) + 1,
                    context=t[max(0, match.start()-230):match.end()+120].replace('\n', ' ')))

visit(R / 'thesis.tex')
source_order = list(dict.fromkeys(e['key'] for e in events))
aux = (R / 'build/current/thesis.aux').read_text(encoding='utf-8')
aux_order = list(dict.fromkeys(k.strip() for group in re.findall(r'\\citation\{([^}]+)\}', aux) for k in group.split(',')))
numbers = {k: int(n) for k,n in re.findall(r'\\bibcite\{([^}]+)\}\{(\d+)\}', aux)}
bbl = (R / 'build/current/thesis.bbl').read_text(encoding='utf-8')
bbl_order = re.findall(r'\\bibitem(?:\[[^]]*\])?\{([^}]+)\}', bbl)
assert set(source_order) <= entries.keys()
assert source_order == aux_order == bbl_order, (source_order, aux_order, bbl_order)
assert numbers == {k:n+1 for n,k in enumerate(bbl_order)}, numbers
for p in active:
    for group in re.findall(r'\\cite\w*(?:\[[^]]*\])*\{([^}]+)\}', uncomment(p.read_text(encoding='utf-8'))):
        ordered = [numbers[k.strip()] for k in group.split(',')]
        assert ordered == sorted(ordered) or '\\usepackage[nocompress]{cite}' in (R/'thesis.tex').read_text(), (p, group, ordered)
assert not any(entries[k]['type'] == 'unpublished' or k.startswith('shen2026') for k in source_order)

log = (R / 'build/current/thesis.log').read_text(encoding='utf-8', errors='replace')
warnings = [s for s in log.splitlines() if re.search(r'^(?:LaTeX|Package .*) Warning:.*(?:undefined|multiply defined)|^Overfull|Rerun to get', s, re.I)]
texts = {p:uncomment(p.read_text(encoding='utf-8')) for p in active}
labels = [x for t in texts.values() for x in re.findall(r'\\label\{([^}]+)\}',t)]
labels += [x for t in texts.values() for x in re.findall(r'\}\{(fig:[^}]+)\}',t)]
refs = [x for t in texts.values() for x in re.findall(r'\\(?:ref|eqref|pageref|autoref)\{([^}]+)\}',t)]
missing_refs = sorted(set(refs)-set(labels))
duplicate_labels = [k for k,n in Counter(labels).items() if n>1]
figures = [x for t in texts.values() for x in re.findall(r'\\thesisfigure\{([^}]+)\}',t)]
fig_labels = [k for k in labels if k.startswith('fig:')]
figure_callouts_missing = sorted(set(fig_labels)-set(refs))
table_float_positions = [x for t in texts.values() for x in re.findall(r'\\begin\{table\}(?:\[([^]]*)\])?',t)]
hardcoded = [dict(file=p.relative_to(R).as_posix(), literal=m.group(0)) for p,t in texts.items()
             for m in re.finditer(r'\[\d+(?:\s*,\s*\d+)*\]', t)]

doc = pymupdf.open(R/('build/current/thesis.pdf' if '--built' in sys.argv else 'thesis.pdf'))
names = doc.resolve_names()
pdf_links, link_errors = [], []
for page in doc:
    for link in page.get_links():
        destination = link.get('nameddest', '')
        if not destination.startswith('cite.'): continue
        key = destination[5:]
        shown = page.get_textbox(link['from']).strip()
        digits = re.findall(r'\d+', shown)
        expected = numbers.get(key)
        dest_page = link.get('page', -1)
        problems = []
        if digits != [str(expected)]: problems.append('visible_number_mismatch')
        if destination not in names or names[destination]['page'] != dest_page:
            problems.append('destination_mismatch')
        if not 0 <= dest_page < len(doc): problems.append('invalid_target_page')
        elif not re.search(r'\[\s*'+str(expected)+r'\s*\]', doc[dest_page].get_text()):
            problems.append('bibliography_number_absent_on_target_page')
        row = dict(page=page.number+1, key=key, number=expected, shown=shown,
                   bibliography_page=dest_page+1, problems=problems)
        pdf_links.append(row)
        if problems: link_errors.append(row)

paper_inventory = []
for key in source_order:
    if entries[key]['type'] not in ('article','inproceedings'): continue
    p = R / 'bibfiles/papers' / (key+'.pdf')
    item = dict(key=key, local_pdf=p.exists())
    if p.exists():
        d = pymupdf.open(p)
        item.update(pages=len(d), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    paper_inventory.append(item)

diff = []
for p in active + [R/'bibfiles/references.bib']:
    rel = p.relative_to(R)
    before = A/'before'/rel
    if before.exists():
        diff.extend(difflib.unified_diff(before.read_text(encoding='utf-8').splitlines(True),
          p.read_text(encoding='utf-8').splitlines(True), fromfile='before/'+rel.as_posix(), tofile=rel.as_posix()))
(A/'reviewed_changes.diff').write_text(''.join(diff),encoding='utf-8')
numeric_changes = []
for p in active:
    rel=p.relative_to(R); before=A/'before'/rel
    if not before.exists(): continue
    def numeric(t):
        t=re.sub(r'\\cite\w*(?:\[[^]]*\])*\{[^}]+\}', '',t)
        t=re.sub(r'\\(?:label|ref|eqref)\{[^}]+\}', '',t)
        return re.findall(r'(?<![A-Za-z])\d+(?:[.,]\d+)*',t)
    old,new=numeric(before.read_text(encoding='utf-8')),numeric(p.read_text(encoding='utf-8'))
    if old!=new: numeric_changes.append(dict(file=rel.as_posix(),difference=list(difflib.ndiff(old,new))))

report = dict(active_files=len(active),cited_sources=len(source_order),
    paper_sources=len(paper_inventory),official_resources=sum(entries[k]['type']=='misc' for k in source_order),
    work_record_sources=0, source_aux_bbl_order_match=True, continuous_numbering=True,
    citation_groups_ascending=True,citation_mentions=len(events),pdf_citation_links=len(pdf_links),pdf_link_errors=link_errors,
    missing_internal_refs=missing_refs,duplicate_labels=duplicate_labels,build_warnings=warnings,
    chapter_counts=[dict(chapter=c,mentions=sum(e['chapter']==c for e in events),
      distinct_sources=len({e['key'] for e in events if e['chapter']==c})) for c in range(1,7)],
    figures=len(figures),figure_callouts_missing=figure_callouts_missing,
    figure_macro_top='\\begin{figure}[!t]' in texts[R/'thesis.tex'],
    non_top_table_options=[s for s in table_float_positions if 't' not in s],
    appendix_present=any('\\appendix' in t for t in texts.values()),
    hardcoded_bracket_numbers_for_review=hardcoded,numeric_changes=numeric_changes,
    paper_inventory=paper_inventory,pdf_pages=len(doc),
    canonical_matches_build=hashlib.sha256((R/'thesis.pdf').read_bytes()).digest()==hashlib.sha256((R/'build/current/thesis.pdf').read_bytes()).digest())
(A/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
(A/'citation_occurrences.json').write_text(json.dumps(events,ensure_ascii=False,indent=2),encoding='utf-8')
(A/'pdf_citation_links.json').write_text(json.dumps(pdf_links,ensure_ascii=False,indent=2),encoding='utf-8')
with (A/'reference_number_map.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['number','key','type','title','source','citation_mentions'])
    for key in source_order:
        e=entries[key];w.writerow([numbers[key],key,e['type'],e['title'],e.get('url',e.get('howpublished','')),sum(x['key']==key for x in events)])
print(json.dumps({k:v for k,v in report.items() if k not in ('paper_inventory','numeric_changes','hardcoded_bracket_numbers_for_review')},ensure_ascii=False),flush=True)
assert not link_errors and not missing_refs and not duplicate_labels and not warnings
assert not figure_callouts_missing and (report['canonical_matches_build'] or '--built' in sys.argv)
