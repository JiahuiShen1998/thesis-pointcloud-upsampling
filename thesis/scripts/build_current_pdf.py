"""Build the original thesis.tex and update the canonical thesis.pdf."""
from pathlib import Path
import subprocess,shutil,sys
R=Path(__file__).resolve().parents[1]
out=R/'build/current';out.mkdir(parents=True,exist_ok=True)
log=R/'audit/manuscript_revision_20260913/build_current.console.txt'
log.parent.mkdir(parents=True,exist_ok=True)
exe=shutil.which('latexmk')
if not exe:raise SystemExit('latexmk is not on PATH.')
# Root sidecars from a previous direct pdflatex build can shadow outdir files.
# Preserve them in the audit before compiling in the dedicated output directory.
archive=R/'audit/manuscript_revision_20260913/legacy_root_build_files'
archive.mkdir(parents=True,exist_ok=True)
for ext in ('aux','bbl','blg','fdb_latexmk','fls','lof','log','lot','toc','out','synctex.gz'):
 p=R/('thesis.'+ext)
 if p.exists():
  target=archive/p.name
  if target.exists():target=archive/(p.name+'.'+str(p.stat().st_mtime_ns))
  shutil.move(str(p),str(target))
print('Compiling thesis.tex to build/current/thesis.pdf',flush=True)
with log.open('w',encoding='utf-8') as f:
 process=subprocess.run([exe,'-pdf','-outdir=build/current',
                         '-interaction=nonstopmode','-halt-on-error','thesis.tex'],
                        cwd=R,stdout=f,stderr=subprocess.STDOUT)
if process.returncode:
 print(log.read_text(encoding='utf-8',errors='replace')[-5000:])
 raise SystemExit(process.returncode)
shutil.copy2(out/'thesis.pdf',R/'thesis.pdf')
print('Published '+str(R/'thesis.pdf'),flush=True)
