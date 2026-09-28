"""Extract the reviewed chapters from thesis.pdf and verify the requested scope."""
import os
from pathlib import Path
import sys,re,json,hashlib,difflib,runpy,contextlib,io
R=Path(__file__).resolve().parents[1]
A=R/'audit/chapters_1_2_restoration_20260914'
sys.path.insert(0,str(R/'.python-deps'))
import pymupdf
d=pymupdf.open(R/'thesis.pdf');toc=d.get_toc();names=d.resolve_names()
starts={title:page-1 for level,title,page in toc if level==1}
first=starts['Introduction'];second=starts['Fundamentals and Related Work']
third=starts['Idea and Concept'];bib=starts['Bibliography']
selected=list(range(first,third))+list(range(bib,len(d)))
mapping={old:new for new,old in enumerate(selected)}
review=pymupdf.open()
for old in selected:review.insert_pdf(d,from_page=old,to_page=old,links=False,annots=True)
rebuilt=0
for old,new in mapping.items():
    for link in d[old].get_links():
        if link.get('page',-1) in mapping:
            target=link['page'];position=link.get('to',pymupdf.Point(0,0))
            if not isinstance(position,pymupdf.Point):
                v=names.get(link.get('nameddest',''),{})
                position=pymupdf.Point(v.get('to',[0,0]))*d[target].transformation_matrix
            review[new].insert_link({'kind':pymupdf.LINK_GOTO,'from':link['from'],'page':mapping[target],'to':position})
            rebuilt+=1
        elif link.get('kind')==pymupdf.LINK_URI:
            review[new].insert_link({'kind':pymupdf.LINK_URI,'from':link['from'],'uri':link['uri']})
review.set_toc([[level,title,mapping[page-1]+1] for level,title,page in toc if page-1 in mapping])
review.set_metadata({'title':'Chapters 1 and 2 review with bibliography and CV placeholder','author':'Jiahui Shen','subject':'Extract from thesis.pdf; original numbering retained.'})
dest=R/'review_chapters_1_2.pdf';temporary=R/'review_chapters_1_2.pending.pdf'
review.save(temporary,garbage=4,deflate=True);review.close();temporary.replace(dest)
rd=pymupdf.open(dest)
assert rd[0].get_text()==d[first].get_text()
assert 'Curriculum Vitae' in rd[-1].get_text()
link_errors=[]
for page in rd:
    for link in page.get_links():
        if link['kind']==pymupdf.LINK_GOTO and not 0<=link.get('page',-1)<len(rd):
            link_errors.append(dict(page=page.number+1,link=repr(link)))
assert not link_errors
frozen=json.loads((A/'frozen_chapters_3_6.json').read_text(encoding='utf-8'))
assert all(hashlib.sha256((R/'texfiles'/n).read_bytes()).hexdigest()==h for n,h in frozen.items())
intro=(R/'texfiles/01_introduction.tex').read_text(encoding='utf-8')
ch2=(R/'texfiles/02_fundamentals.tex').read_text(encoding='utf-8')
assert not any(x in intro for x in ['\\thesisfigure','\\includegraphics','\\begin{figure}','\\section*'])
assert '\\paragraph*' not in ch2 and 'MAX' not in ch2
assert all(x not in ch2 for x in ['TULIP','SPU-PMD','SPD-PMD'])
assert (R/'texfiles/cv.tex').read_bytes()==(A/'before/texfiles/cv.tex').read_bytes()
receipt=dict(pdf_pages=len(d),chapter_1_physical_pages=[first+1,second],
 chapter_2_physical_pages=[second+1,third],review_pages=len(rd),
 review_original_pages=[p+1 for p in selected],rebuilt_internal_links=rebuilt,
 review_link_errors=link_errors,chapter2_equations=ch2.count('\\begin{equation}'),
 chapter2_figures=ch2.count('\\thesisfigure'),introduction_figures=0,
 introduction_small_headings=0,chapter2_paragraph_headings=0,
 max_operators_lowercase=True,cv_restored_placeholder=True,chapters_3_6_unchanged=True,
 canonical_sha256=hashlib.sha256((R/'thesis.pdf').read_bytes()).hexdigest(),
 review_sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
(A/'final_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
S=Path(os.environ.get('RESEARCH_REVIEW_HELPERS_DIR', Path(__file__).resolve().parent/'external_review_helpers'));sys.path.insert(0,str(S))
for p in [R/'thesis.pdf',dest]:
    sys.argv=[str(S/'pdf_read_preflight.py'),str(p)]
    with contextlib.redirect_stdout(io.StringIO()) as stream:
        try:runpy.run_path(str(S/'pdf_read_preflight.py'),run_name='__main__')
        except SystemExit:pass
    result=json.loads(stream.getvalue())
    (A/(p.stem+'.preflight.json')).write_text(json.dumps(result,indent=2))
    assert result['verdict']=='PASS',result
for n in ['01_introduction.tex','02_fundamentals.tex']:
    original=(A/('archived_'+n)).read_text(encoding='utf-8-sig')
    revised=(R/'texfiles'/n).read_text(encoding='utf-8')
    (A/(n+'.original.diff')).write_text(''.join(difflib.unified_diff(original.splitlines(True),revised.splitlines(True),fromfile='original/'+n,tofile='revised/'+n)),encoding='utf-8')
qadir=A/'pages_final';qadir.mkdir(exist_ok=True)
for i in list(range(first,third))+[len(d)-1]:
    d[i].get_pixmap(matrix=pymupdf.Matrix(1.1,1.1),alpha=False).save(qadir/f'page_{i+1}.png')
report=(A/'REPORT_CN.md').read_text(encoding='utf-8')
report=re.sub(r' The original project is a complete compilation of documents. \d+  Page ',f' The original project is a complete compilation of documents. {len(d)}  Page ',report)
report=re.sub(r' Chapters I and II review copies, \d+  Page ',f' Chapters I and II review copies, {len(rd)}  Page ',report)
report=re.sub(r' to the extent of the reservation  \d+  Internal links ',f' to the extent of the reservation  {rebuilt}  Internal links ',report)
report=re.sub(r' The first chapter is in the whole book.  PDF  I don’t think so.  .*?CV  I don’t think so.  \d+  Pages. ',f' The first chapter is in the whole book.  PDF  I don’t think so.  {first+1}–{second}  I, resolution 1, annex II.  {second+1}–{third}  Page CV  I don’t think so.  {len(d)}  Pages. ',report)
report=re.sub(r' Compare with blanks, original  .*? Quality indicators. ',' The names above are no longer shown as subheadings in the order in which the most recent paragraph content is requested to be retained. ',report)
report=re.sub(r' Table  2\.1  is located in the body of Chapter II  \d+  At the top of the page, it just started.  2\.2  classification. ',' Table  2.1  At the top of the corresponding page and keep at  2.2  classification Before the text. ',report)
report=report.replace('pages/','pages_final/')
if '##  Recent subheading amendments ' not in report:
    report+='\n##  Recent subheading amendments \n\n Delete directly in the original chapter file  Introduction  It’s...  6  Subheadings, chapter II  18  Subheadings within paragraphs; necessary chapter numbering and method name retained. 5  Location  MAX  Operator to lowercase  max,  Do not change the meaning of the formula. See the corresponding record  heading_correction.json.  This round maintains the main body of chapters III to VI unchanged. \n'
(A/'REPORT_CN.md').write_text(report,encoding='utf-8')
print(json.dumps(receipt),flush=True)

