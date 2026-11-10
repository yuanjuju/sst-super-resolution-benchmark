import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const root=process.argv[2];
const out=process.argv[3];
if(!root || !out) throw new Error('Usage: node build_workbook.mjs REPOSITORY OUTPUT_DIRECTORY');
const data=JSON.parse(await fs.readFile(path.join(root,'tables/natural_image_sr/workbook_source.json'),'utf8'));
const wb=Workbook.create();
const names=['FFT对照','CAMixerSR','全部指标','Set5逐图','本地推理'];
const sheets=names.map(n=>wb.worksheets.add(n));
function base(sh,range){
  sh.showGridLines=false;
  sh.getRange(range).format.font={name:'Arial',size:11,color:'#18243B'};
  sh.getRange(range).format.rowHeight=23;
  sh.getRange(range).format.verticalAlignment='center';
}
function title(sh,text,end){
  sh.getRange('B2').values=[[text]];
  sh.getRange('B2').format.font={name:'Arial',size:16,bold:true,color:'#18243B'};
  sh.getRange(`B3:${end}3`).format.borders={bottom:{style:'thin',color:'#A7B5C9'}};
}
function table(sh,headers,rows,start=5){
  const col=String.fromCharCode(65+headers.length);
  sh.getRange(`B${start}:${col}${start}`).values=[headers];
  sh.getRange(`B${start}:${col}${start}`).format={fill:'#263D63',font:{name:'Arial',size:11,bold:true,color:'#FFFFFF'},horizontalAlignment:'center',rowHeight:28};
  if(rows.length) sh.getRangeByIndexes(start,1,rows.length,headers.length).values=rows;
  for(let i=0;i<rows.length;i++) if(i%2===0) sh.getRangeByIndexes(start+i,1,1,headers.length).format.fill='#F1F4F8';
  return start+rows.length;
}
function line(sh,range,titleText,from,to,colors){
  const c=sh.charts.add('line',sh.getRange(range));
  c.title=titleText;c.titleTextStyle.fontSize=13;c.titleTextStyle.typeface='Arial';
  c.setPosition(from,to);
  c.legend={position:'top',textStyle:{typeface:'Arial',fontSize:11}};
  c.xAxis={axisType:'textAxis',textStyle:{typeface:'Arial',fontSize:11}};
  c.yAxis={numberFormatCode:'0.00',numberFormatSourceLinked:false,textStyle:{typeface:'Arial',fontSize:11}};
  for(let i=0;i<c.series.items.length;i++) c.series.items[i].line={fill:colors[i%colors.length],style:'solid',width:2};
  return c;
}

const fft=sheets[0];base(fft,'B2:N30');title(fft,'SAFMN FFT损失对照','N');
fft.getRange('B4').values=[['RTX5090 / batch64 / seed10 / DIV2K val100 / EMA / RGB / crop4']];
const steps=[5000,10000,15000,20000,25000,30000];
const paired=steps.map(step=>[step,...['fft','nofft'].map(arm=>data.training.find(r=>r.experiment==='SAFMN_pair_5090_'+arm && r.step===step).psnr_rgb_db),null]);
table(fft,['更新步数','L1 + FFT / dB','L1 / dB','FFT增益 / dB'],paired,6);
for(let i=7;i<=12;i++) fft.getRange(`E${i}`).formulas=[[`=C${i}-D${i}`]];
fft.getRange('C7:E12').setNumberFormat('0.0000');
fft.getRange('B6:B12').format.columnWidth=15;fft.getRange('C6:E12').format.columnWidth=20;
fft.getRange('G:N').format.columnWidth=10;
line(fft,'B6:D12','DIV2K验证PSNR（dB）','G6','O22',['#2563A5','#CE6A2B']);
const meanRows=data.set5_mean.map(r=>[r.arm,Number(r.iter),Number(r.psnr_rgb_db),Number(r.psnr_y_db),Number(r.ssim_y)]);
table(fft,['损失组','更新步数','Set5 RGB / dB','Set5 Y / dB','Set5 Y SSIM'],meanRows,16);
fft.getRange('D17:F20').setNumberFormat('0.000000');fft.getRange('F16:F20').format.columnWidth=19;
fft.getRange('B24').values=[['来源：foundations/natural-image-sr/results/safmn_fft_pair_30k/results/']];

const cam=sheets[1];base(cam,'B2:N25');title(cam,'CAMixerSR 5000步训练','N');
cam.getRange('B4').values=[['RTX5090 / batch16 / seed10 / L1 + 路由约束 / DIV2K EMA RGB / crop4']];
const camRows=data.training.filter(r=>r.model==='CAMixerSR' && r.kind==='scheduled').map(r=>[r.step,r.psnr_rgb_db]);
table(cam,['更新步数','验证PSNR / dB'],camRows,6);
cam.getRange('B:C').format.columnWidth=24;cam.getRange('C7:C11').setNumberFormat('0.0000');
cam.getRange('B14').values=[['结束重复验证 / dB']];cam.getRange('C14').values=[[27.5540]];cam.getRange('C14').setNumberFormat('0.0000');
cam.getRange('E:N').format.columnWidth=10;
line(cam,'B6:C11','CAMixerSR验证PSNR（dB）','E6','N22',['#307C67']);
cam.getRange('B24').values=[['来源：foundations/natural-image-sr/results/camixersr5000.result.json']];

const all=sheets[2];base(all,'B2:J46');title(all,'全部测试指标','J');
all.getRange('B4').values=[['保留各次测试及重复验证；数据集、颜色通道和权重键分别列出。']];
const headers=['模型','检查点','权重键','数据集','通道','裁边','PSNR / dB','SSIM','原记录说明'];
const rows=data.metrics.map(r=>[r.model,r.checkpoint,r.weight_key,r.dataset,r.channel,Number(r.crop_border),Number(r.psnr_db),r.ssim?Number(r.ssim):null,r.notes]);
table(all,headers,rows);all.freezePanes.freezeRows(5);
all.getRange('B:B').format.columnWidth=15;all.getRange('C:C').format.columnWidth=34;
all.getRange('D:E').format.columnWidth=20;all.getRange('F:G').format.columnWidth=10;
all.getRange('H:I').format.columnWidth=16;all.getRange('J:J').format.columnWidth=95;
all.getRange('H6:I46').setNumberFormat('0.0000');

const per=sheets[3];base(per,'B2:K25');title(per,'Set5逐图结果','K');
per.getRange('B4').values=[['五张图 × 两组损失 × 5k/30k；EMA、裁边4。']];
table(per,['损失组','更新步数','图像','RGB PSNR / dB','RGB SSIM','Y PSNR / dB','Y SSIM','LR宽','LR高','预测文件'],data.set5.map(r=>[r.arm,Number(r.iter),r.image,Number(r.psnr_rgb_db),Number(r.ssim_rgb),Number(r.psnr_y_db),Number(r.ssim_y),Number(r.input_width),Number(r.input_height),r.prediction_path]));
per.getRange('B:D').format.columnWidth=17;per.getRange('E:H').format.columnWidth=19;
per.getRange('I:J').format.columnWidth=12;per.getRange('K:K').format.columnWidth=100;
per.getRange('E6:H25').setNumberFormat('0.000000');per.freezePanes.freezeRows(5);

const local=sheets[4];base(local,'B2:I15');title(local,'本地CPU单图推理','I');
local.getRange('B4').values=[['SAFMN早期30k权重 / RGB / crop4 / Pillow bicubic对照']];
table(local,['样例','权重键','SR PSNR / dB','Pillow PSNR / dB','增益 / dB','单次耗时 / s','设备','来源'],data.local.map(r=>[r.run,r.weight_key,r.psnr_rgb_db,r.pillow_bicubic_psnr_rgb_db,null,r.wall_seconds,r.device,r.source]));
for(let i=6;i<=8;i++)local.getRange(`F${i}`).formulas=[[`=D${i}-E${i}`]];
local.getRange('B:C').format.columnWidth=24;local.getRange('D:G').format.columnWidth=21;local.getRange('H:H').format.columnWidth=12;local.getRange('I:I').format.columnWidth=95;
local.getRange('D6:G8').setNumberFormat('0.0000');

wb.recalculate();
await fs.mkdir(out,{recursive:true});
console.log((await wb.inspect({kind:'table',range:'FFT对照!B6:E12',include:'values,formulas',tableMaxRows:8,tableMaxCols:4,maxChars:2800})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!',options:{useRegex:true,maxResults:20},maxChars:1500})).ndjson);
const output=await SpreadsheetFile.exportXlsx(wb);await output.save(path.join(out,'experiments.xlsx'));
for(let i=0;i<sheets.length;i++){
  const range=['B2:O24','B2:N24','B2:I17','B2:J16','B2:H11'][i];
  const render=await wb.render({sheetName:names[i],range,scale:1.5,format:'png'});
  await fs.writeFile(path.join(out,`sheet-${i+1}.png`),new Uint8Array(await render.arrayBuffer()));
}
console.log(JSON.stringify({sheets:names,output:path.join(out,'experiments.xlsx')}));
