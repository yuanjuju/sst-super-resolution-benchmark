"""Show both pre-existing parameter variants, never select a favorable checkpoint."""
import datetime
import hashlib
import json
import os
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=HERE/'visualizations_20261110'
os.environ['MPLCONFIGDIR']=str(HERE/'.cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager


def main():
    font=Path('/root/wmsr-repro/strong_models/candidate_screen_20261109/.cache/fonts/STHeiti-Medium.ttc')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),
                         'axes.unicode_minus':False,'font.size':11,'svg.fonttype':'path','savefig.facecolor':'white'})
    verification=json.loads((HERE/'records/final_verification.json').read_text())
    assert verification['status']=='pass'
    histories={n:[json.loads(s) for s in (HERE/f'pilot/results/{n}_OISST_x4_s10/validation_history.jsonl').read_text().splitlines()] for n in ['ESC','WMSR']}
    bicubic=json.loads((HERE/'records/bicubic_development80.json').read_text())['metrics']
    colors={'ESC':'#c55a11','WMSR':'#1768ac'}
    fig,axs=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    names=['ESC','WMSR']
    for ax,key,title,unit in [(axs[0,0],'psnr_y_db','5000步末点 PSNR（越高越好）','dB'),
                             (axs[0,1],'rmse_c','5000步末点温度RMSE（越低越好）','℃')]:
        vmax=0.
        for variant,offset,color in [('raw',-.18,'#496987'),('ema',.18,'#c4a066')]:
            values=[histories[n][-1][variant][key] for n in names];vmax=max(vmax,max(values))
            bars=ax.bar([i+offset for i in range(2)],values,width=.34,color=color,label='raw 普通权重' if variant=='raw' else 'EMA 平滑权重')
            for b,v in zip(bars,values):
                ax.annotate(f'{v:.2f}' if key=='psnr_y_db' else f'{v:.3f}',xy=(b.get_x()+b.get_width()/2,v),xytext=(0,4),textcoords='offset points',ha='center',fontsize=10)
        ax.axhline(bicubic[key],linestyle=':',color='#777777',label='Bicubic')
        ax.set(xticks=[0,1],xticklabels=names,ylabel=unit,title=title,ylim=(0,vmax*1.2))
        ax.legend(fontsize=9,loc='lower right' if key=='psnr_y_db' else 'upper right')
        ax.spines[['top','right']].set_visible(False)
    for ax,key,title,label in [(axs[1,0],'psnr_y_db','完整验证PSNR曲线','PSNR / dB'),
                              (axs[1,1],'rmse_c','完整验证温度RMSE曲线（对数轴）','RMSE / ℃')]:
        for name in names:
            h=histories[name]
            for variant,style in [('raw','-'),('ema','--')]:
                ax.plot([v['step'] for v in h],[v[variant][key] for v in h],style,color=colors[name],linewidth=1.7,marker='o' if variant=='raw' else None,markersize=3,label=f'{name} {variant}')
        ax.axhline(bicubic[key],color='#777777',linestyle=':',label='Bicubic')
        ax.axvline(1000,color='#aaaaaa',linewidth=.8)
        ax.set(title=title,xlabel='累计参数更新次数（1000步后为本轮续训）',ylabel=label)
        if key=='rmse_c':ax.set_yscale('log')
        ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.suptitle('OISST ×4｜同一80张训练内验证图｜同时报告raw与EMA\n固定5000步末点，不是原200张测试集成绩',fontsize=16)
    for ext in ['png','svg']:fig.savefig(OUT/f'04_raw_ema_comparison.{ext}',dpi=180)
    plt.close(fig)
    raw=json.loads((OUT/'visualization_manifest.json').read_text())
    ema=json.loads((OUT/'ema/visualization_manifest.json').read_text())
    assert [s['sample_id'] for s in raw['selected_samples']]==[s['sample_id'] for s in ema['selected_samples']]
    assert raw['checkpoint_sha256']==ema['checkpoint_sha256']
    delta={kind:{k:histories['ESC'][-1][kind][k]-histories['WMSR'][-1][kind][k] for k in ['psnr_y_db','rmse_c','mae_c','ssim_y']} for kind in ['raw','ema']}
    analysis={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'population':'training-only development80; not original200test',
              'fixed_updates':5000,'endpoint_selection':'prescribed5000 for both raw and EMA; no replacement with best intermediate checkpoint',
              'ESC_minus_WMSR':delta,'raw_ESC_last_two':histories['ESC'][-2:],'no_new_optimizer_updates':True,
              'same_four_visual_ids_raw_and_ema':True,'raw_error_color_range':raw['error_color_range_shared'],'ema_error_color_range':ema['error_color_range_shared']}
    (HERE/'records/raw_ema_analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
    report=HERE/'5000步对照结果.md'
    text=report.read_text().split('\n## raw与EMA的结果解释')[0]
    text+='\n## raw与EMA的结果解释\n\n'
    text+=f'固定5000步，raw下ESC相对WMSR的PSNR差为{delta["raw"]["psnr_y_db"]:+.4f} dB，RMSE差为{delta["raw"]["rmse_c"]:+.5f} ℃；EMA下PSNR差为{delta["ema"]["psnr_y_db"]:+.4f} dB，RMSE差为{delta["ema"]["rmse_c"]:+.5f} ℃。本轮raw由WMSR占优，EMA下ESC的PSNR与温度RMSE占优，但WMSR的SSIM略高，结论需要注明参数版本及指标。不能只挑有利的一列宣布胜出。\n\n'
    text+='横看本轮固定末点的全部记录，ESC的EMA为53.5225 dB / 0.11703 ℃，WMSR的raw为54.0029 dB / 0.11589 ℃；ESC仍低0.4804 dB，温度RMSE高约0.00114 ℃。这是不同参数版本间的描述性差异，不能用来变更预先规定的raw/EMA对照或代替正式测试。\n\n'
    text+='ESC的raw PSNR在4800步为53.9811 dB，5000步为45.9947 dB；同一时期EMA从53.4404升至53.5225 dB。最后100步的平均训练L1由前一窗口0.001963升至0.003489，表明末段存在波动，但未出现NaN，核验不支持将其称为训练程序失败。我们保留这一末点和整条曲线，不自动延长、改学习率或替换为中间高分。\n\n'
    text+='EMA是训练中对权重做指数平滑的另一套参数，不是另训练一个网络。两模型使用相同衰减系数.999，但各自的原生随机初始化仍不同；这些短预算结果不证明充分收敛。原200张正式测试仍未使用。\n\n'
    text+='总览见visualizations_20261110/04_raw_ema_comparison.png；根目录其他图为raw；ema子目录是同样4个冻结样本的EMA重建和误差图。每张图都给出实际单位和色标；逐图重算误差已与对应80图验证记录核对。\n'
    report.write_text(text)
    (OUT/'看图说明.md').write_text('# 5000步结果看图说明\n\n优先看04_raw_ema_comparison.png：柱图比较固定5000步，曲线包含全部25次验证。raw是当步普通权重，EMA是相同训练中积累的平滑权重。PSNR越高越好、RMSE越低越好；右下RMSE曲线使用对数轴。\n\n根目录02、03为raw的温度重建与误差；ema目录是同4个样本的EMA版本。样本固定为验证集前4个，未按结果挑选。温度图每行共用色标，误差图注明共同色标和4像素边界裁剪。\n\n这是训练内80张验证图，不能当作原200张正式测试成绩或直接与论文表格排名。\n')
    (OUT/'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>ESC与WMSR5000步结果</title><style>body{max-width:1300px;margin:30px auto;padding:0 20px;font:17px/1.7 sans-serif;color:#243445}img{width:100%;height:auto;margin:12px 0 28px}h1,h2{font-weight:600}</style><h1>ESC与WMSR：固定5000步</h1><p>同一OISST×4训练内80张验证图。原200张正式测试集未使用。raw和EMA都完整展示，不挑最佳中间步。</p><h2>两种参数的末点和完整曲线</h2><img src="04_raw_ema_comparison.png"><h2>raw：同一4个固定样本的重建</h2><img src="02_temperature_reconstructions.png"><img src="03_signed_temperature_errors.png"><h2>EMA：同样4个样本</h2><img src="ema/02_temperature_reconstructions.png"><img src="ema/03_signed_temperature_errors.png"></html>''')
    files={str(f.relative_to(OUT)):{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in OUT.rglob('*') if f.is_file() and f.name!='complete_figures_manifest.json'}
    (OUT/'complete_figures_manifest.json').write_text(json.dumps({'at':analysis['at'],'files':files,'parameter_variants':['raw','ema'],'same4_fixed_validation_ids':True},indent=2)+'\n')
    print(json.dumps({'status':'complete','parameter_variants':['raw','ema'],'delta':delta,'figures':len(files)}))


if __name__=='__main__':main()
