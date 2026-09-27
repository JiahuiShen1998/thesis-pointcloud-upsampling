"""Check manuscript tables and descriptive analyses against retained observations."""
from pathlib import Path
import csv,json,re,collections,statistics
R=Path(__file__).resolve().parents[1];M=R/'evidence/modelnet40_hpc/data'
D=R/'evidence/kitti_lab/visual_audit/reports/dual_detector_three_frame_root_cause_20260730/analysis'
Q=R/'audit/results_expansion_20260913/qa'
def rows(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
errors=[];class_values=0
source=(R/'texfiles/05_modelnet_class_tables.tex').read_text(encoding='utf-8')
sections=re.split(r'\\paragraph\{Line ([AB])\.\}',source)
methods=['EAR','PDANS','PU-Net','PU-GCN','PU-EdgeFormer']
for i in range(1,len(sections),2):
 line,body=sections[i:i+2];wide=rows(M/f'classification_per_class_line{line}_wide.csv')
 baseline=next(k for k in wide[0] if k not in methods+['class_idx','class_name','n_test'])
 for text in body.splitlines():
  if not re.match(r'^[a-z][a-z ]+\s*&',text):continue
  cells=[x.strip() for x in text.removesuffix(r'\\').split('&')]
  rr=next(x for x in wide if x['class_name'].replace('_',' ')==cells[0])
  for method,value in zip([baseline]+methods,cells[1:]):
   expected=f'{float(rr[method]):.1f}';class_values+=1
   if value!=expected:errors.append(['category',line,cells[0],method,value,expected])
# Recompute paired geometry using the complete paired object exports.
geometry=rows(M/'geometry_equal_n_per_sample.csv')
paired=json.loads((R/'audit/results_expansion_20260913/derived_statistics.json').read_text(encoding='utf-8'))['paired_geometry']
for expected in paired:
 line=expected['line'];metric=expected['metric']
 a={x['shape_id']:float(x[metric]) for x in geometry if x['line']==line and x['display']=='PU-GCN'}
 b={x['shape_id']:float(x[metric]) for x in geometry if x['line']==line and x['display']=='PU-Net'}
 assert len(a)==len(b)==2468 and a.keys()==b.keys()
 delta=[a[k]-b[k] for k in a]
 for key,value in [('n',len(delta)),('gcn_lower',sum(x<0 for x in delta)),('net_lower',sum(x>0 for x in delta)),('median',statistics.median(delta))]:
  if abs(value-expected[key])>1e-12:errors.append(['paired',line,metric,key,value,expected[key]])
# Reaggregate all retained GT events; this matching diagnostic is not official AP.
groups=collections.defaultdict(collections.Counter);event_rows=0
with (D/'all_frame_gt_transitions.csv').open(encoding='utf-8-sig',newline='') as f:
 for x in csv.DictReader(f):
  key=tuple(x[k] for k in ['detector','line','method','class'])
  groups[key][x['transition']]+=1;event_rows+=1
events=['maintained','confidence_degraded','localization_degraded','lost_after_upsampling','recovered_after_upsampling','missed_both']
summary=rows(D/'transition_summary_full_val.csv')
for x in summary:
 key=tuple(x[k] for k in ['detector','line','method','class']);c=groups[key]
 derived={event:c[event] for event in events}
 derived['gt_opportunities']=sum(c.values())
 derived['baseline_tp']=sum(c[e] for e in events[:4])
 derived['upsampled_tp']=sum(c[e] for e in events[:3])+c['recovered_after_upsampling']
 derived['net_tp_change']=derived['upsampled_tp']-derived['baseline_tp']
 for name,value in derived.items():
  if value!=int(x[name]):errors.append(['event',*key,name,value,x[name]])
counts=json.loads((Q/'effective_input_counts.json').read_text(encoding='utf-8'))
assert len(counts)==96 and all(x['equal'] for x in counts)
result={'modelnet_category_values':class_values,'modelnet_paired_comparisons':len(paired),'objects_per_pair':2468,'kitti_event_rows':event_rows,'kitti_event_groups':len(summary),'case_input_counts_checked':len(counts),'errors':errors}
(R/'audit/manuscript_revision_20260913/qa/evidence_numerical_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2),flush=True)
assert class_values==480
assert not errors,errors
