"""大型Q2复用artifact生成的Q1样式，保持全部数值，避免整表对象化。"""
from pathlib import Path
import zipfile,re,xml.etree.ElementTree as ET
B=Path(__file__).resolve().parent.parent/'A题_改进交付版/提交结果'
with zipfile.ZipFile(B/'result1.xlsx') as z:
    ref=z.read('xl/worksheets/sheet1.xml');styles=z.read('xl/styles.xml')
ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'};root=ET.fromstring(ref)
cells={c.attrib['r']:c for c in root.findall('.//s:c',ns)}
styleA=cells['A2'].attrib['s'].encode();styleB=cells['B2'].attrib['s'].encode()
head=re.search(rb'<x:row r="1".*?</x:row>',ref).group().replace(b'<x:',b'<').replace(b'</x:',b'</')
prefix=re.search(rb'<x:sheetViews>.*?</x:cols>',ref).group().replace(b'<x:',b'<').replace(b'</x:',b'</')
source=B/'result2.xlsx';target=B/'result2_styled.tmp.xlsx'
with zipfile.ZipFile(source) as z,zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as out:
    for info in z.infolist():
        data=z.read(info.filename)
        if info.filename=='xl/styles.xml':data=styles
        if info.filename.startswith('xl/worksheets/sheet') and info.filename.endswith('.xml'):
            data=re.sub(rb'<sheetViews>.*?<sheetData>',prefix+b'<sheetData>',data,count=1)
            cut=data.index(b'</row>')+len(b'</row>');before=data[:data.index(b'<row r="1"')]+head
            rest=data[cut:]
            rest=re.sub(rb'<c r="([A-Z]+)([0-9]+)"',lambda m:b'<c r="'+m[1]+m[2]+b'" s="'+(styleA if m[1]==b'A' else styleB)+b'"',rest)
            data=before+rest
            data=data.replace(b'<sheetViews>',b'<dimension ref="A1:V205811"/><sheetViews>',1)
        out.writestr(info,data)
target.replace(source)
print('Q2已统一表头、四位小数显示、列宽和冻结窗格')
