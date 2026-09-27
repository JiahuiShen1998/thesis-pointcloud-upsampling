"""Verify SHA256SUMS; --refresh explicitly regenerates it after intentional edits."""
from pathlib import Path
import hashlib,sys
root=Path(__file__).resolve().parents[1]
manifest=root/'SHA256SUMS'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def keep(p):
 rel=p.relative_to(root)
 return not any(x in {'.git','__pycache__','build','audit','.venv','.python-deps'} for x in rel.parts) and p!=manifest and rel.as_posix()!='thesis/thesis.pdf' and not p.name.endswith(('.pyc','.pyo'))
if sys.argv[1:]==['--refresh']:
 files=sorted(p for p in root.rglob('*') if p.is_file() and keep(p))
 manifest.write_text(''.join(f'{digest(p)}  {p.relative_to(root).as_posix()}\n' for p in files),encoding='utf-8')
 print(f'Updated SHA256SUMS: {len(files)} files')
elif sys.argv[1:]:
 raise SystemExit('Usage: python tools/check_archive.py [--refresh]')
else:
 failed=[];count=0
 for line in manifest.read_text(encoding='utf-8').splitlines():
  expected,name=line.split('  ',1);p=root/name;count+=1
  if not p.is_file() or digest(p)!=expected:failed.append(name)
 if failed:
  print('Missing or changed files:\n'+'\n'.join(failed));raise SystemExit(1)
 print(f'PASS: {count} files match SHA256SUMS')
