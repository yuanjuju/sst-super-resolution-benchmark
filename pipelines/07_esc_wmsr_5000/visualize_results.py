#!/usr/bin/env python3
"""Research figures from frozen development IDs; no training/test200 selection."""
from pathlib import Path
import datetime
import hashlib
import json
import math
import os
import sys

HERE=Path(__file__).resolve().parent
BASE=Path('/root/wmsr-repro')
OUT=HERE/'visualizations_20261110'
os.environ['MPLCONFIGDIR']=str(HERE/'.cache/matplotlib')
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter,MaxNLocator
from train_continue import make_model,batch_tensors
from factory_esc import cache_rpe_indices


class WMSR01(torch.nn.Module):
    def __init__(self,network):super().__init__();self.network=network
    def forward(self,x):return self.network(x*255.)/255.


def configure():
    OUT.mkdir(exist_ok=True)
    font=Path('/root/wmsr-repro/strong_models/candidate_screen_20261109/.cache/fonts/STHeiti-Medium.ttc')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),
                         'axes.unicode_minus':False,'font.size':11,'axes.titleweight':'normal',
                         'savefig.facecolor':'white','svg.fonttype':'path'})
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False


def long_label(x,_=None):
    v=(x+180)%360-180
    return f'{abs(v):.0f}°'+('E' if v>=0 else 'W')


def lat_label(x,_=None):return f'{abs(x):.0f}°'+('N' if x>=0 else 'S')


def geo_axes(ax):
    ax.xaxis.set_major_locator(MaxNLocator(3));ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.xaxis.set_major_formatter(FuncFormatter(long_label));ax.yaxis.set_major_formatter(FuncFormatter(lat_label))
    ax.tick_params(labelsize=8)


def main():
    configure()
    protocol=json.loads((HERE/'protocol.json').read_text())
    split=json.loads((HERE/'development_splits/OISST.json').read_text())
    manifest_path=BASE/'records/formal/OISST_manifest.json'
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==split['source_manifest_sha256']
    data=json.loads(manifest_path.read_text())
    indices=split['validation_indices'][:4]
    metas=[data['splits']['train']['samples'][i] for i in indices]
    assert [m['sample_id'] for m in metas]==split['validation_ids'][:4]
    with np.load(BASE/data['splits']['train']['file']) as a:
        high_c=a['hr_celsius'].copy();low_c=a['lr_x4_celsius'].copy()
    lr=((low_c+2)/42).copy();hr=((high_c+2)/42).copy()
    low,high=batch_tensors(lr,hr,indices)
    esc=cache_rpe_indices(make_model('ESC',4).cuda(),'cuda').eval()
    esc_path=HERE/'pilot/results/ESC_OISST_x4_s10/checkpoints/pilot_final.pt'
    esc_state=torch.load(esc_path,map_location='cpu',weights_only=True);assert esc_state['step']==5000
    esc.load_state_dict(esc_state['model']);del esc_state
    with torch.inference_mode():esc_output=torch.cat([esc(low[i:i+1]) for i in range(4)])
    esc_c=(esc_output.mean(1)*42-2).cpu().numpy()
    del esc,esc_output
    torch.cuda.empty_cache()
    sys.path.insert(0,str(BASE))
    from extended_models.factory import create_model
    raw,_=create_model({'model':'WMSR','scale':4,'channels':64})
    wmsr=WMSR01(raw).cuda().eval()
    reference=HERE/'pilot/results/WMSR_OISST_x4_s10'
    w_state=torch.load(reference/'checkpoints/pilot_final.pt',map_location='cpu',weights_only=True);assert w_state['step']==5000
    wmsr.load_state_dict(w_state['model']);del w_state
    with torch.inference_mode():
        w_output=torch.cat([wmsr(low[i:i+1]) for i in range(4)])
        bicubic=torch.nn.functional.interpolate(low,scale_factor=4,mode='bicubic',align_corners=False,antialias=False)
    w_c=(w_output.mean(1)*42-2).cpu().numpy();b_c=(bicubic.mean(1)*42-2).cpu().numpy()
    del wmsr,w_output,bicubic,low,high
    torch.cuda.empty_cache()
    truth=high_c[indices];lows=low_c[indices]
    maps={'Bicubic':b_c,'WMSR':w_c,'ESC':esc_c}
    errors={name:(value-truth)[:,4:-4,4:-4] for name,value in maps.items()}
    canonical={
        'Bicubic':json.loads((HERE/'records/bicubic_development80.json').read_text()),
        'WMSR':json.loads((reference/'validation_005000_raw.json').read_text()),
        'ESC':json.loads((HERE/'pilot/results/ESC_OISST_x4_s10/validation_005000_raw.json').read_text())}
    selected=[]
    for j,meta in enumerate(metas):
        row={'sample_id':meta['sample_id'],'original_train_index':indices[j],'validation_position':j,'date':meta['date'],
             'lat_bounds':meta['lat_bounds'],'lon_bounds':meta['lon_bounds'],'metrics':{}}
        for name in maps:
            existing=next(r for r in canonical[name]['samples'] if r['sample_id']==meta['sample_id'])
            measured=float(np.sqrt(np.mean(errors[name][j].astype(np.float64)**2)))
            expected=math.sqrt(existing['mse_c2'])
            assert abs(measured-expected)<1e-4,(meta['sample_id'],name,measured,expected)
            row['metrics'][name]={'rmse_c':expected,'visual_recomputed_rmse_c':measured}
        selected.append(row)
    np.savez_compressed(OUT/'selected4_reconstructions.npz',sample_id=np.array([m['sample_id'] for m in metas]),
                        hr_celsius=truth,lr_x4_celsius=lows,bicubic_celsius=b_c,wmsr_celsius=w_c,esc_celsius=esc_c)
    # Both bar axes start at zero; learning curves show their explicit scale.
    esc_history=[json.loads(s) for s in (HERE/'pilot/results/ESC_OISST_x4_s10/validation_history.jsonl').read_text().splitlines()]
    w_history=[json.loads(s) for s in (reference/'validation_history.jsonl').read_text().splitlines()]
    summaries={'Bicubic':canonical['Bicubic']['metrics'],
               'WMSR':canonical['WMSR']['summary'],'ESC':canonical['ESC']['summary']}
    colors={'Bicubic':'#707f8f','WMSR':'#1768ac','ESC':'#c55a11'}
    fig,axs=plt.subplots(2,2,figsize=(11,7.5),constrained_layout=True)
    names=['Bicubic','WMSR','ESC']
    for ax,key,title,unit in [(axs[0,0],'psnr_y_db','5000步末点：PSNR（越高越好）','dB'),(axs[0,1],'rmse_c','5000步末点：温度RMSE（越低越好）','℃')]:
        values=[summaries[n][key] for n in names]
        ax.bar(names,values,color=[colors[n] for n in names],width=.6)
        ax.set(title=title,ylabel=unit,ylim=(0,max(values)*1.15))
        for i,v in enumerate(values):ax.text(i,v+max(values)*.02,f'{v:.2f}' if key=='psnr_y_db' else f'{v:.3f}',ha='center')
        ax.spines[['top','right']].set_visible(False)
    for ax,key,title,label in [(axs[1,0],'psnr_y_db','验证PSNR学习曲线','PSNR / dB'),(axs[1,1],'rmse_c','验证温度RMSE学习曲线','RMSE / ℃')]:
        for name,hist in [('WMSR',w_history),('ESC',esc_history)]:
            ax.plot([r['step'] for r in hist],[r['raw'][key] for r in hist],marker='o',color=colors[name],label=name)
        ax.axhline(summaries['Bicubic'][key],color=colors['Bicubic'],linestyle=':',label='Bicubic')
        ax.set(title=title,xlabel='参数更新次数',ylabel=label);ax.grid(alpha=.2);ax.legend()
    fig.suptitle('OISST ×4｜同一80张训练内验证图｜raw参数\n不是原200张正式测试集成绩',fontsize=15)
    for ext in ['png','svg']:fig.savefig(OUT/f'01_metrics_and_curves.{ext}',dpi=180)
    plt.close(fig)
    # Preserve native pixel values; use a common temperature color scale within each row.
    fig,axs=plt.subplots(4,5,figsize=(14,11),constrained_layout=True)
    titles=['低分辨率输入 24×24','高分辨率真值 96×96','Bicubic插值','WMSR（5000步）','ESC（5000步）']
    extents=[];temperature_ranges=[]
    for j,meta in enumerate(metas):
        extent=[meta['lon_bounds'][0]-.125,meta['lon_bounds'][1]+.125,meta['lat_bounds'][0]-.125,meta['lat_bounds'][1]+.125]
        extents.append(extent)
        fields=[lows[j],truth[j],b_c[j],w_c[j],esc_c[j]]
        vmin=min(float(x.min()) for x in fields);vmax=max(float(x.max()) for x in fields)
        temperature_ranges.append([vmin,vmax])
        for k,field in enumerate(fields):
            im=axs[j,k].imshow(field,origin='lower',extent=extent,cmap='viridis',vmin=vmin,vmax=vmax,interpolation='nearest',aspect='equal')
            geo_axes(axs[j,k])
            if j==0:axs[j,k].set_title(titles[k],fontsize=12)
            if k==0:axs[j,k].set_ylabel(f'样本{j+1}  {meta["date"]}',fontsize=10)
        fig.colorbar(im,ax=list(axs[j,:]),shrink=.8,pad=.01,label='海表温度 / ℃')
    fig.suptitle('同一海域的输入、真值与重建海温图\n固定取验证集前4个样本；每行共用色标；两模型均为raw5000步',fontsize=16)
    fig.savefig(OUT/'02_temperature_reconstructions.png',dpi=180)
    plt.close(fig)
    # Fixed shared error scale across all examples/methods, no percentile clipping.
    error_limit=max(float(np.max(np.abs(v))) for v in errors.values())
    fig,axs=plt.subplots(4,3,figsize=(10,11),constrained_layout=True)
    for j,meta in enumerate(metas):
        full=extents[j];extent=[full[0]+1,full[1]-1,full[2]+1,full[3]-1]
        for k,name in enumerate(['Bicubic','WMSR','ESC']):
            im=axs[j,k].imshow(errors[name][j],origin='lower',extent=extent,cmap='RdBu_r',vmin=-error_limit,vmax=error_limit,interpolation='nearest',aspect='equal')
            geo_axes(axs[j,k]);rmse=selected[j]['metrics'][name]['rmse_c']
            axs[j,k].set_title(f'{name} − 真值\nRMSE={rmse:.3f}℃',fontsize=11)
            if k==0:axs[j,k].set_ylabel(f'样本{j+1}  {meta["date"]}',fontsize=10)
    fig.colorbar(im,ax=list(axs.flat),shrink=.7,pad=.02,label='温度误差 / ℃（红：偏暖；蓝：偏冷）')
    fig.suptitle('重建误差热图｜越接近白色，误差越小\n与指标一致剔除4像素边界；全部图使用同一误差色标',fontsize=15)
    fig.savefig(OUT/'03_signed_temperature_errors.png',dpi=180)
    plt.close(fig)
    evidence={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'dataset':'OISST','scale':4,
              'evaluation_population':'Frozen80-image training-only validation, not final200 test',
              'selection_rule':'First4 validation IDs in frozen order, chosen before prediction inspection',
              'selected_samples':selected,'temperature_color_ranges_per_row':temperature_ranges,
              'error_color_range_shared':[-error_limit,error_limit],'error_crop_border':4,
              'temperature_map_scalar':'Unclipped inverse[-2,40]C mapping of mean3 predicted channels, same as physical evaluation',
              'orientation':'NetCDF lat ascending and unflipped crop; origin lower. Extent includes half native0.25degree grid spacing.',
              'checkpoint_sha256':{'ESC':hashlib.sha256(esc_path.read_bytes()).hexdigest(),'WMSR':hashlib.sha256((reference/'checkpoints/pilot_final.pt').read_bytes()).hexdigest()},
              'metric_summaries_80':summaries,'training_updates_per_model':5000,'new_optimizer_updates_for_visuals':0,
              'files':{f.name:{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in OUT.iterdir() if f.is_file() and f.name in ['01_metrics_and_curves.png','01_metrics_and_curves.svg','02_temperature_reconstructions.png','03_signed_temperature_errors.png','selected4_reconstructions.npz']}}
    (OUT/'visualization_manifest.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'status':'complete','output':str(OUT),'samples':[m['sample_id'] for m in metas],'shared_error_limit_c':error_limit}),flush=True)


if __name__=='__main__':main()
