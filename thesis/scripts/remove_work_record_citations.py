"""Remove the eight author work-record bibliography entries from the thesis.

Experimental evidence stays in the local evidence and audit directories.
This script has no network access and does not modify experimental data.
"""
from pathlib import Path
import re, json, shutil

R = Path(__file__).resolve().parents[1]
A = R/'audit/remove_work_record_citations_20260914'
A.mkdir(parents=True, exist_ok=True)
KEYS = {
    'shen2026modelnet', 'shen2026kitti', 'shen2026kittifinal',
    'shen2026kittivis', 'shen2026modelnetupdated', 'shen2026kittiupdated',
    'shen2026kittiprimary', 'shen2026geometryaudit',
}

def backup(p):
    q = A/'before'/p.relative_to(R)
    q.parent.mkdir(parents=True, exist_ok=True)
    if not q.exists():
        shutil.copy2(p,q)

seen = set()
def walk(p):
    if p in seen: return
    seen.add(p)
    t = re.sub(r'(?<!\\)%[^\n]*', '', p.read_text(encoding='utf-8'))
    for v in re.findall(r'\\(?:input|include|tableinput)\{([^}]+)\}',t):
        q = R/v
        walk(q if q.suffix else q.with_suffix('.tex'))

walk(R/'thesis.tex')
targets = seen | {R/'scripts/generate_record_tables.py', R/'scripts/check_package.py'}
changes = []
pattern = re.compile(r'[ \t~]*\\cite\w*(?:\[[^]]*\])*\{([^}]+)\}')
for p in sorted(targets):
    t = p.read_text(encoding='utf-8')
    matches = []
    def replace(m):
        keys = [k.strip() for k in m.group(1).split(',')]
        removed = [k for k in keys if k in KEYS]
        if not removed: return m.group()
        kept = [k for k in keys if k not in KEYS]
        matches.append(dict(line=t.count('\n',0,m.start())+1,removed=removed,kept=kept))
        if kept:
            return m.group().replace(m.group(1),','.join(kept))
        return ''
    result = pattern.sub(replace,t)
    if matches:
        result = re.sub(r'\n[ \t]*([.,;:])',r'\1',result)
        result = re.sub(r'([.!?])[ \t]+\n',r'\1\n',result)
        result = re.sub(r'\n(?:[ \t]*\n){2,}','\n\n',result)
        backup(p)
        p.write_text(result,encoding='utf-8')
        changes.append(dict(file=p.relative_to(R).as_posix(),citations=matches))

p = R/'bibfiles/references.bib'
t = p.read_text(encoding='utf-8')
removed_entries = []
for m in reversed(list(re.finditer(r'@\w+\s*\{\s*([^,]+),',t))):
    if m.group(1).strip() not in KEYS: continue
    start = t.index('{',m.start())
    depth = 1
    end = start+1
    while depth:
        if t[end]=='{' and t[end-1]!='\\': depth+=1
        if t[end]=='}' and t[end-1]!='\\': depth-=1
        end+=1
    removed_entries.append(t[m.start():end])
    t = t[:m.start()]+t[end:]
if removed_entries:
    backup(p)
    t = t.replace('%%%% Primary records of experiments performed for this thesis %%%%','%%%% Official benchmark evaluation rules %%%%')
    t = re.sub(r'\n{3,}','\n\n',t).rstrip()+'\n'
    p.write_text(t,encoding='utf-8')
    (A/'removed_entries.txt').write_text('\n\n'.join(reversed(removed_entries))+'\n',encoding='utf-8')
report=dict(removed_entries=len(removed_entries),files_changed=changes,active_files=len(seen),remaining_record_keys_in_active_sources=[str(p.relative_to(R)) for p in seen if any(k in p.read_text(encoding='utf-8') for k in KEYS)])
(A/'removal.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(dict(removed_entries=report['removed_entries'],changed_files=len(changes),citation_locations=sum(len(c['citations']) for c in changes),remaining=report['remaining_record_keys_in_active_sources']),indent=2))
