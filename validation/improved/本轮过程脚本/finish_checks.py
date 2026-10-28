from solve import *
import solve as s
import shutil,zipfile,re
BASE=ROOT.parent/'A题_改进交付版';OLD=ROOT.parent/'A题_终稿'
diag=[]
for n in (640,1280,2560):
    m=Model(4,n,tail='last',mesh='surface');sol=m.solve(120,method='Radau')
    T,C=sample(m,sol,[60.,120.],np.arange(21)/10);v=sol.sol([60.,120.])
    diag.append({'n':n,'T_surface':v[m.n].tolist(),'C_surface':v[2*(m.n+1)-1].tolist()})
(BASE/'验证/Q4早期加密.json').write_text(json.dumps(diag,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(diag,ensure_ascii=False),flush=True)

# Artifact exporter exhausted 8 GB on this full workbook earlier. Preserve the existing
# package and patch only changed numerical cells using the standard OOXML container.
# This is not a new full-workbook writer; unaffected XML/styles remain byte-identical.
def letters(j):
    ans=''
    while j:j,k=divmod(j-1,26);ans=chr(65+k)+ans
    return ans
changes={}
for sheet,field in ((1,'T'),(2,'C')):
    a=np.loadtxt(OLD/'计算结果'/f'q2_{field}.csv',delimiter=',',skiprows=1)
    b=np.loadtxt(BASE/'计算结果'/f'q2_{field}.csv',delimiter=',',skiprows=1)
    ii,jj=np.where(a!=b)
    changes[f'xl/worksheets/sheet{sheet}.xml']={f'{letters(int(j)+1)}{int(i)+2}':format(float(b[i,j]),'.12g') for i,j in zip(ii,jj)}
    del a,b
book=BASE/'提交结果/result2.xlsx';temp=book.with_suffix('.tmp.xlsx')
with zipfile.ZipFile(book) as zin,zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED) as zout:
    for info in zin.infolist():
        content=zin.read(info.filename)
        if info.filename in changes:
            edits=[]
            for cell,value in changes[info.filename].items():
                pos=content.index(('r="'+cell+'"').encode());end=content.index(b'</c>',pos)
                start=content.index(b'<v>',pos)+3;stop=content.index(b'</v>',start)
                assert start<stop<end
                edits.append((start,stop,value.encode()))
            for start,stop,value in sorted(edits,reverse=True):content=content[:start]+value+content[stop:]
        zout.writestr(info,content)
temp.replace(book)
(BASE/'验证/Q2定点修订.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2),encoding='utf-8')
print('Q2定点更新完成',flush=True)

# Export remaining small official tables with the standard artifact tool.
shutil.copy2(ROOT/'export_results.mjs',BASE/'export_results.mjs')
shutil.copy2(ROOT/'verify_exports.py',BASE/'verify_exports.py')
# Match the source template's numeric radius headings, not its CSV unit annotation.
# Reproducible solver: relative data path works from the code subfolder.
src=(ROOT/'solve.py').read_text(encoding='utf-8')
src=src.replace("ROOT = Path(__file__).resolve().parent","ROOT = Path(__file__).resolve().parent.parent")
src=src.replace("tail='nominal'","tail='last'")
src=src.replace("cases=[args.case] if args.case else [1,2,3,4]","cases=[args.case] if args.case else [1,2,4]")
src=src.replace("model.case=q\n        print", "if q==2: save_case(model,sol)\n        model.case=q\n        print")
(BASE/'求解代码/solve.py').write_text(src,encoding='utf-8')
shutil.copy2(ROOT/'资料复核改进/环境情景统一配置.json',BASE/'验证/环境情景统一配置.json')
shutil.copy2(ROOT/'资料复核改进/显式符号恒等核验.json',BASE/'验证/显式符号恒等核验.json')
shutil.copy2(ROOT/'审查复跑/表面加密对比.json',BASE/'验证/历史表面加密对比.json')
