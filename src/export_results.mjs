import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';
const here=path.dirname(fileURLToPath(import.meta.url));      // …/A题_终稿/求解代码
const calc=path.join(here,'..','计算结果');
const out=path.join(here,'..','提交结果');
await fs.mkdir(out,{recursive:true});
const q=Number(process.argv[2]||1);
const wb=Workbook.create();
const fields=q<=2?[['温度','T'],['水分浓度','C']]:[['Sheet1','C']];
for (const [name,field] of fields){
 const txt=(await fs.readFile(path.join(calc,`q${q}_${field}.csv`),'utf8')).replace(/^\uFEFF/,'').trim();
 const lines=txt.split(/\r?\n/);const header=lines.shift().split(',');
 header[0]='时间\\到药材中心的距离';
 const data=[header.map((v,i)=>i>0&&v!=='药材表面'?Number(v):v),...lines.map(l=>l.split(',').map(v=>v===''?null:Number(v)))];
 const sh=wb.worksheets.add(name);const nr=data.length,nc=header.length;
 for(let k=0;k<nr;k+=1000)sh.getRangeByIndexes(k,0,Math.min(1000,nr-k),nc).values=data.slice(k,k+1000);
 const used=sh.getRangeByIndexes(0,0,nr,nc);
 used.format.font.name='Microsoft YaHei';used.format.font.size=10;
 used.setNumberFormat('0.0000');
 sh.getRangeByIndexes(0,1,1,nc-1).setNumberFormat('0.0');
 sh.getRangeByIndexes(1,0,nr-1,1).setNumberFormat('0');
 sh.getRangeByIndexes(0,0,1,nc).format.fill='#E8EEF5';
 sh.getRangeByIndexes(0,0,1,nc).format.font.bold=true;
 sh.getRangeByIndexes(0,0,nr,1).format.columnWidth=31;
 sh.getRangeByIndexes(0,1,nr,nc-1).format.columnWidth=11;
 sh.getRangeByIndexes(0,0,1,nc).format.rowHeight=29;
 sh.freezePanes.freezeRows(1);sh.freezePanes.freezeColumns(1);
 sh.showGridLines=true;
 console.log(JSON.stringify({q,sheet:name,rows:nr,cols:nc}));
}
wb.recalculate();
for(const [name] of fields){
 try{
  const image=await wb.render({sheetName:name,range:'A1:G10',scale:1.5,format:'png'});
  await fs.writeFile(path.join(out,`预览_q${q}_${name}.png`),new Uint8Array(await image.arrayBuffer()));
 }catch(e){console.warn('preview skipped:',e.message);}
}
const output=await SpreadsheetFile.exportXlsx(wb);
await output.save(path.join(out,`result${q}.xlsx`));
console.log(`saved result${q}.xlsx`);
