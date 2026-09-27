"""Extract the reviewed chapters from thesis.pdf and verify the requested scope."""
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
S=Path.home()/'.codex/skills/academic-research-suite/ars/scripts';sys.path.insert(0,str(S))
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
report=re.sub(r'原工程完整编译文件，\d+ 页',f'原工程完整编译文件，{len(d)} 页',report)
report=re.sub(r'第一、二章审阅副本，\d+ 页',f'第一、二章审阅副本，{len(rd)} 页',report)
report=re.sub(r'保留范围内的 \d+ 个内部链接',f'保留范围内的 {rebuilt} 个内部链接',report)
report=re.sub(r'第一章在整本 PDF 第 .*?CV 第 \d+ 页。',f'第一章在整本 PDF 第 {first+1}–{second} 页，第二章第 {second+1}–{third} 页，CV 第 {len(d)} 页。',report)
report=re.sub(r'以空白分词对照，原稿 .*?质量指标。','按最新要求保留段落内容顺序，以上名称不再显示为小标题。',report)
report=re.sub(r'表 2\.1 位于第二章正文第 \d+ 页顶部，其下才开始 2\.2 分类。','表 2.1 位于相应页面顶部，并保持在 2.2 分类正文之前。',report)
report=report.replace('pages/','pages_final/')
if '## 最新小标题修正' not in report:
    report+='\n## 最新小标题修正\n\n直接在原始章节文件删除 Introduction 的 6 个小标题、第二章的 18 个段内小标题；保留必要的章节编号和方法名称。5 处 MAX 运算符改为小写 max，不改动公式含义。对应记录见 heading_correction.json。本轮保持第三至六章正文不变。\n'
(A/'REPORT_CN.md').write_text(report,encoding='utf-8')
print(json.dumps(receipt),flush=True)

