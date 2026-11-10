#!/usr/bin/env python3
"""Report released-code runs with all outcomes and an explicit reproduction boundary."""
import csv,datetime as dt,hashlib,json,subprocess,tarfile
from pathlib import Path
ROOT=Path('/root/wmsr-repro')

def main():
    subprocess.check_call([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/15_verify_full_datasets.py')])
    subprocess.check_call([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/20_plot_public_examples.py')])
    out=ROOT/'results/public_summary';out.mkdir(exist_ok=True)
    rows=[];checkpoints=[]
    for name in ['HYCOM','OISST','GHRSST']:
        for scale in [2,3,4]:
            run=ROOT/'results/public_control'/f'{name}_x{scale}_seed10'
            result=json.loads((run/'final_results.json').read_text());cfg=result['config']
            assert result['status']=='complete' and result['epochs']==100 and result['optimizer_steps']==10000
            assert cfg['train_count']==800 and cfg['test_count']==200
            assert cfg['architecture']['implementation']=='unmodified pinned author network'
            model=result['test_wmsr'];bicubic=result['test_bicubic']
            assert model['count']==200 and bicubic['count']==200
            rows.append({'dataset':name,'scale':scale,'epochs':100,'optimizer_steps':10000,
                         'parameters':result['parameters'],'model_psnr_y_db':model['psnr_y_db'],
                         'bicubic_psnr_y_db':bicubic['psnr_y_db'],'delta_psnr_y_db':result['delta_psnr_y_db'],
                         'model_ssim_y':model['ssim_y'],'bicubic_ssim_y':bicubic['ssim_y'],
                         'model_rmse_c':model['rmse_c'],'bicubic_rmse_c':bicubic['rmse_c']})
            checkpoints.append({'dataset':name,'scale':scale,'sha256':hashlib.sha256((run/'checkpoints/final.pt').read_bytes()).hexdigest()})
    with (out/'metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    now=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()
    wins=sum(row['delta_psnr_y_db']>0 for row in rows)
    lines=['# WMSR 公开源码：全量数据实验','',f'完成记录时间：{now}','',
           '使用固定提交 dbb1cff4c4b30375928c6ee62224cb4b37fa337a 的未修改模型；实际为4个WAM，与论文24个WAM的描述存在差异。',
           '本轮是公开源码在重建数据上的运行记录，不等于作者论文实验的严格数值复现。',
           '每个数据集1000张HR，800训练/200测试，三个倍率均100epochs、10000次参数更新、batch8、seed10。','',
           '| 数据集 | 倍率 | 公开网络PSNR(Y) | Bicubic PSNR(Y) | 差值dB | 网络SSIM(Y) | 网络RMSE °C | Bicubic RMSE °C |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| '+row['dataset']+' | ×'+str(row['scale'])+' | '+format(row['model_psnr_y_db'],'.4f')+' | '+format(row['bicubic_psnr_y_db'],'.4f')+' | '+format(row['delta_psnr_y_db'],'+.4f')+' | '+format(row['model_ssim_y'],'.6f')+' | '+format(row['model_rmse_c'],'.6f')+' | '+format(row['bicubic_rmse_c'],'.6f')+' |')
    lines+=['',f'本协议下有{wins}/9组PSNR(Y)高于Bicubic；所有结果均保留，不按测试分数选择模型或重跑种子。','',
            '## 与第一轮结构重建的关系','',
            '第一轮按论文补齐未公开结构的近似重建，9/9组低于Bicubic；这是失败结果，单独保留。云端路径为 results/summary 和 results/formal；本地为上层“云端实验结果/results/”目录，本阶段为“云端实验结果/公开源码实验/”，不将两轮混合。',
            '第二轮保持同一数据、划分、训练配置和固定种子，仅使用作者发布的原网络及其初始化。结构与初始化同时不同，因此不能将差值归因于某一个模块。','',
            '## 协议与限制','',
            '- 数据来自公开原始SST：HYCOM原生3小时2016年、OISST2016全年、MUR2016-08-12；各1000张不同内容，来源和裁剪坐标见 ../../records/formal/。',
            '- 以公开loader方式将96解释为HR patch；温度固定[-2,40]映射至0..255、重复三通道；作者原始RGB/伪彩色映射不详。',
            '- LR为PyTorch bicubic antialias=True；Bicubic回放antialias=False。特别是原生MUR非常平滑，基线PSNR很高，不能与论文未知RGB处理的PSNR直接比较。',
            '- 原生HR的数值范围、逐图标准差与相邻像素温差统计见 ../../records/formal/field_statistics.json；这些是描述统计，不能单凭它们证明模型落后基线的原因。',
            '- Adam lr1e-3，beta=(0.9,0.999)，eps1e-8；每20epoch减半，loss=0.1L1+0.9FFL(alpha1)。论文还提及cosine，存在歧义。',
            '- 每epoch完整遍历800张一次，repeat1；公开默认repeat20与此更新数不同，未宣称恢复作者实际训练预算。',
            '- 使用固定最终epoch100权重，测试200张；不按测试结果选checkpoint。Y指标裁剪倍率像素边界，逐图PSNR/SSIM平均；温度RMSE从未裁值的三输出通道均值反归一化计算。',
            '- 单一种子和人工重建的样本划分，只能支持本协议下的实测结论，不能说明论文完整消融或跨年份泛化。','',
            '## 重要文件','',
            '- metrics.csv：完整结果；training_curves.png：训练曲线。',
            '- examples_数据集_x倍率.png：固定测试序号0/66/133的温度与误差图，未按分数挑图。',
            '- ../public_control/：每组配置、源码副本、训练记录、逐图测试分数和final.pt。',
            '- ../../records/WMSR_upstream_dbb1cff4c4b3.tar.gz：固定提交的完整上游源码。',
            '- 环境、原始数据和可恢复last.pt继续保留云端，本地仅回收关键文件。','']
    (out/'README.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12.6,3.7),layout='constrained')
    for ax,scale in zip(axes,[2,3,4]):
        for name in ['HYCOM','OISST','GHRSST']:
            with (ROOT/'results/public_control'/f'{name}_x{scale}_seed10/training_metrics.csv').open() as stream:data=list(csv.DictReader(stream))
            ax.plot([int(x['epoch']) for x in data],[float(x['loss']) for x in data],label=name)
        ax.set_title('Scale x'+str(scale));ax.set_xlabel('Epoch');ax.set_yscale('log');ax.grid(alpha=.2)
    axes[0].set_ylabel('0.1 L1 + 0.9 FFL');axes[-1].legend();fig.suptitle('Released WMSR network / full datasets / seed10')
    fig.savefig(out/'training_curves.png',dpi=170);plt.close(fig)
    verification={'checked_at':now,'jobs':9,'all_epochs100':True,'all_optimizer_steps10000':True,
                  'all_train800_test200':True,'all_final_checkpoints_present':True,'checkpoints':checkpoints,
                  'reproduction_boundary':'Unmodified published network, reconstructed SST data and explicit training interpretation'}
    (out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    commit=subprocess.check_output(['git','-C',str(ROOT/'upstream'),'rev-parse','HEAD'],text=True).strip()
    assert commit=='dbb1cff4c4b30375928c6ee62224cb4b37fa337a'
    assert not subprocess.check_output(['git','-C',str(ROOT/'upstream'),'diff','HEAD','--','models'],text=True)
    allowed=[]
    for pattern in ['scripts/*.py','scripts/*.sh','paper_models/*.py','configs/*.txt','records/*.json','records/*.md',
                    'records/environment.lock.txt','records/WMSR_upstream_*.tar.gz','records/formal/*.json','results/public_summary/*',
                    'results/public_control/*/config.json','results/public_control/*/model_source.py',
                    'results/public_control/*/final_results.json','results/public_control/*/training_metrics.csv',
                    'results/public_control/*/*test_per_sample.csv','results/public_control/*/checkpoints/final.pt']:
        allowed.extend(p for p in ROOT.glob(pattern) if p.is_file() and not p.is_symlink())
    bundle=ROOT/'results/WMSR_public_results.tar.gz'
    with tarfile.open(bundle,'w:gz',compresslevel=5) as archive:
        for path in sorted(set(allowed)):archive.add(path,arcname=str(path.relative_to(ROOT)),recursive=False)
    digest=hashlib.sha256(bundle.read_bytes()).hexdigest()
    path=ROOT/'records/public_results_package.json';temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'status':'complete','path':str(bundle),'bytes':bundle.stat().st_size,
                                    'sha256':digest,'packaged_files':len(set(allowed))},indent=2)+'\n');temporary.replace(path)
    print('PUBLIC_RESULTS_PACKAGED',bundle,bundle.stat().st_size,digest,flush=True)

if __name__=='__main__':main()
