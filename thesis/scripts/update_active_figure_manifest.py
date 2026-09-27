"""Record active figure dependencies with portable paths and checksums."""
from pathlib import Path
import re,json,hashlib
R=Path(__file__).resolve().parents[1];O=R/'figure';Q=R/'audit/manuscript_revision_20260913/qa'
old=json.loads((O/'active_manifest.json').read_text());new=json.loads((Q/'new_figure_manifest.json').read_text());meta={r['name']:r for r in old+new}
revision=R/'audit/teacher_email_revision_20260917/figure_manifest.json'
if revision.exists():
 for item in json.loads(revision.read_text(encoding='utf-8')):
  name=item['name'];meta[name].update(item)
  meta[name]['plotting_script']='scripts/make_teacher_revision_figures.py'
seen=set();active=[]
def walk(p):
 if p in seen:return
 seen.add(p);text=p.read_text(encoding='utf-8')
 for name in re.findall(r'\\thesisfigure\{([^}]+)\}',text):active.append((name,p))
 for name in re.findall(r'\\(?:input|include|tableinput)\{([^}]+)\}',text):
  q=R/name;walk(q if q.suffix else q.with_suffix('.tex'))
walk(R/'thesis.tex');result=[]
for name,p in active:
 row=dict(meta[name]);row['source_tex']=p.relative_to(R).as_posix();row['files']={}
 for ext in ['pdf','svg','png']:
  f=O/(name+'.'+ext);assert f.exists(),f
  row['files'][ext]={'file':f.relative_to(R).as_posix(),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
 result.append(row)
(O/'active_manifest.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
print('Active figures:',len(result),'Chapter 5:',sum(x['source_tex'].split('/')[-1].startswith('05_') for x in result))
