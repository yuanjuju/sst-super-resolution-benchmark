#!/usr/bin/env python3
"""Verify full coverage, retain negative findings, and package important files only."""
import argparse,csv,datetime as dt,hashlib,json,subprocess,tarfile
from pathlib import Path
ROOT=Path('/root/wmsr-repro')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--allow-partial',action='store_true');args=parser.parse_args()
    plan=json.loads((ROOT/'records/extended_plan.json').read_text());results={}
    queue=json.loads((ROOT/'records/extended_queue_status.json').read_text())
    failures={x['id']:x for x in queue['failed']}
    assert not queue['active'] and not queue['pending'],'Do not finalize an active queue'
    if failures:assert args.allow_partial,'Failures must be explicitly reported, never called full completion'
    for spec in plan['jobs']:
        run=ROOT/'results/extended'/spec['id'];path=run/'final_results.json'
        if not path.exists():
            assert args.allow_partial and spec['id'] in failures,'Missing result is not a recorded failure'
            continue
        result=json.loads(path.read_text())
        assert result['status']=='complete' and result['epochs']==spec['epochs'] and result['optimizer_steps']==spec['epochs']*100
        assert result['test_model']['count']==200 and result['config']['train_count']==800 and result['config']['test_count']==200
        assert hashlib.sha256((run/'checkpoints/final.pt').read_bytes()).hexdigest()==result['checkpoint_sha256']
        with (run/'training_metrics.csv').open() as f:metrics=list(csv.DictReader(f))
        assert [int(x['epoch']) for x in metrics]==list(range(1,spec['epochs']+1))
        results[spec['id']]=result
    assert set(results).isdisjoint(failures) and len(results)+len(failures)==len(plan['jobs'])
    subprocess.run([str(ROOT/'.venv5090/bin/python'),str(ROOT/'scripts/42_profile_extended.py')],check=True,cwd=ROOT)
    plot_command=[str(ROOT/'.venv5090/bin/python'),str(ROOT/'scripts/43_plot_extended.py')]
    if failures:plot_command.append('--allow-missing')
    subprocess.run(plot_command,check=True,cwd=ROOT)
    out=ROOT/'results/extended_summary';out.mkdir(exist_ok=True);profiles={x['key']:x for x in json.loads((out/'profiles.json').read_text())}
    rows=[];regions=[]
    for identifier,result in results.items():
        spec=result['spec'];m=result['test_model'];b=result['test_bicubic'];profile=profiles[spec['preflight_key']]
        rows.append({'id':identifier,'kind':spec['kind'],'method':spec['model'],'dataset':spec['dataset'],'scale':spec['scale'],
            'epochs':result['epochs'],'steps':result['optimizer_steps'],'parameters':result['parameters'],
            'psnr_y_db':m['psnr_y_db'],'ssim_y':m['ssim_y'],'rmse_c':m['rmse_c'],'mae_c':m['mae_c'],
            'bicubic_psnr_y_db':b['psnr_y_db'],'delta_psnr_y_db':result['delta_psnr_y_db'],
            'estimated_flops':profile['estimated_total_flops'],'inference_ms_batch1':profile['inference_ms_batch1']})
        for region,values in result['per_region'].items():regions.append({'id':identifier,'region':region,**values['model']})
    def write_csv(path,items):
        if not items:return
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(items[0]));writer.writeheader();writer.writerows(items)
    write_csv(out/'all_metrics.csv',rows);write_csv(out/'hycom_regions.csv',regions)
    write_csv(out/'failed_experiments.csv',[{'id':k,'reason':v['tail'].splitlines()[-1],'log':v['log']} for k,v in failures.items()])
    (out/'failures.json').write_text(json.dumps(list(failures.values()),indent=2)+'\n')
    for kind in ['comparison','ablation','channel_sweep','group_sweep','loss_sweep']:
        subset=[dict(x,alias_of='') for x in rows if x['kind']==kind]
        for alias,target in plan['aliases'].items():
            alias_kind='channel_sweep' if alias.startswith('channels_') else 'loss_sweep'
            if alias_kind==kind and target in results:
                row=next(x for x in rows if x['id']==target)
                subset.append(dict(row,id=alias,kind=kind,alias_of=target))
        write_csv(out/(kind+'_metrics.csv'),subset)
    methods=['EDSR','ELAN','DAT','SAFMN','MambaIR','CRAN','WMSR'];now=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()
    lines=['# WMSR 论文补充实验：执行结果与中断记录','',f'汇总时间：{now}','',
        f"预设{len(plan['jobs'])}个独立任务，本轮{len(results)}组完整训练及测试完成，{len(failures)}组中断。中断任务没有完整测试分数，不计为成功完成。",
        '计划包括63组方法比较（六种比较方法及5090的WMSR参照）、9组20epochs消融、4组新增通道、6组组/块数、4组新增损失。两个重复点引用对应主实验。',
        '成功任务均沿用原800训练/200测试，不缩减数据或batch。消融20epochs/2000更新，其余100epochs/10000更新，seed10，固定最终epoch权重。','',
        '这份记录覆盖预设实验的执行状态，但不代表所有任务成功，也不是作者私有实验版本的严格恢复。中断和负结果全部保留，未调整模型/参数来重跑失败任务。','']
    for dataset in ['HYCOM','OISST','GHRSST']:
        lines+=['## '+dataset+'方法比较','','| 方法 | ×2 PSNR/SSIM | ×3 PSNR/SSIM | ×4 PSNR/SSIM |','|---|---:|---:|---:|']
        for method in methods:
            cells=[]
            for scale in [2,3,4]:
                identifier=f'compare_{method}_{dataset}_x{scale}_s10'
                if identifier not in results:cells.append('中断，无完整结果');continue
                m=results[identifier]['test_model'];cells.append(f"{m['psnr_y_db']:.4f} / {m['ssim_y']:.6f}")
            lines.append('| '+method+('*' if method=='CRAN' else '')+' | '+' | '.join(cells)+' |')
        lines.append('')
    lines+=['## 模型、消融与比较边界','',
        '- 比较模型采用固定公开来源：EDSR r16f64、ELAN-light、DAT类默认结构、SAFMN-L(128/16)、MambaIR官方lightSR配置。它们不全部等于WMSR论文中未公开的比较模型变体；实测参数与来源见配置及all_metrics.csv。',
        '- CRAN*：按原CRAN论文section3.5采用10组、每组19普通残差块+1CRAB；Eq1–2的CDRR生效，CGC实现参考固定第三方及原CGC源码。它是标明来源的论文指导重建，并非官方发布网络或已确认的WMSR比较变体。',
        '- 方法比较中六种比较方法用L1，WMSR使用0.1L1+0.9FFL；统一数据/预算/优化器，不将其解释为仅架构不同的因果实验。',
        '- 消融基于发布版4-WAM网络；所有Case包括基线均训练20epochs。移除DWT采用全分辨率低/高分支，PDC改为C通道并取分支均值融合。这些融合细节未获作者确认，与原TableVI不同实现的数值不直接比较。',
        '- 组数实验用公开WAM显式组装RWAMG（组内3×3细化和组残差），不使用首轮失败的32通道瓶颈重建，也不以参数量接近原表作为正确性证明。',
        '- 通道扫描32/48/64/96/128；组/块扫描2/4/6组×2/4块；Fig4损失系数按PDF目视核对为0/.1/.5/.9/1。64通道及0.1L1的重复点引用WMSR GHRSST ×4主实验。',
        '- 温度编码、HR96定义、bicubic抗混叠、repeat1和学习率歧义与前轮一致；测试图仅用于预设最终评测和报告敏感性，不据测试结果挑checkpoint或重跑种子。单种子敏感性结果不证明泛化或稳定性。',
        '- 本轮独立5090环境使用Torch2.7.1+cu128及本项目sm120编译的Mamba2.2.5；旧4090结果独立保留。blackwell_acceptance.json记录CUDA/reference梯度与旧权重200张回放的一致性检查。','',
        '## 可复查文件','',
        '- all_metrics.csv、hycom_regions.csv：全部结果及HYCOM四区域结果。',
        '- failed_experiments.csv、failures.json：中断任务、原始错误和日志路径；失败日志也保留在归档logs/中。',
        '- comparison/ablation/channel_sweep/group_sweep/loss_sweep_metrics.csv：按实验类别分表。',
        '- profiles.json：实测参数、统一HR96输入下的近似FLOPs和batch1延迟。FLOPs为工具支持的卷积/矩阵乘加上显式SSM算术估计，未计入所有非线性/指数等运算，不是原论文工具的精确恢复。',
        '- comparison_*.png与errors_*.png：固定测试序号0/66/133的全部方法对比，原温度/误差图使用统一色标。',
        '- ../extended/：每组配置、训练CSV、逐图指标及final.pt；../../configs/extended：预设实验规格；../../extended_models：冻结架构与明确的适配。',
        '- 原始数据、环境、CUDA编译工具链留在云端；完成实验的优化器last.pt已转为紧凑final.pt；旧实验没有删除。','',
        '依据：[WMSR论文](https://arxiv.org/abs/2509.24334)、[WMSR源码](https://github.com/oucailab/WMSR)、[CRAN论文](https://donglaiw.github.io/paper/2021_iccv_sr_context.pdf)。','']
    (out/'README.md').write_text('\n'.join(lines))
    verification={'checked_at':now,'status':'partial' if failures else 'complete','jobs':len(results),'planned_jobs':len(plan['jobs']),
        'failed_jobs':len(failures),'failed_ids':list(failures),'all_planned_complete':not failures,
        'verification_scope':'Successfully completed jobs only; missing failed checkpoints are not claimed verified',
        'all_expected_epochs':True,'all_train800_test200':True,
        'all_optimizer_steps_verified':True,'all_final_checkpoints_verified':True,'aliases':plan['aliases'],
        'checkpoints':[{'id':k,'sha256':v['checkpoint_sha256']} for k,v in results.items()]}
    (out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    allowed=[]
    patterns=['scripts/*.py','scripts/*.sh','extended_models/**/*.py','configs/*.txt','configs/extended/*.json',
        'records/*.json','records/*.txt','records/*.patch','records/WMSR_upstream_*.tar.gz','records/formal/*.json','records/extended_ready/*.json',
        'external/ELAN/configs/*.yml','external/MambaIR/options/train/mambair/*lightSR*.yml','external/*/LICENSE*',
        'results/extended_summary/*','results/extended/*/config.json','results/extended/*/final_results.json',
        'results/extended/*/training_metrics.csv','results/extended/*/*test_per_sample.csv','results/extended/*/checkpoints/final.pt']
    for pattern in patterns:allowed.extend(p for p in ROOT.glob(pattern) if p.is_file() and not p.is_symlink())
    allowed.extend(ROOT/'logs'/('extended_'+identifier+'.log') for identifier in failures)
    bundle=ROOT/'results/WMSR_extended_results.tar.gz'
    with tarfile.open(bundle,'w:gz',compresslevel=3) as archive:
        for path in sorted(set(allowed)):archive.add(path,arcname=str(path.relative_to(ROOT)),recursive=False)
    digest=hashlib.sha256(bundle.read_bytes()).hexdigest();target=ROOT/'records/extended_results_package.json';temp=target.with_suffix('.tmp')
    temp.write_text(json.dumps({'status':'partial' if failures else 'complete','completed_jobs':len(results),'failed_jobs':len(failures),
        'planned_jobs':len(plan['jobs']),'path':str(bundle),'bytes':bundle.stat().st_size,'sha256':digest,'files':len(set(allowed))},indent=2)+'\n');temp.replace(target)
    print('EXTENDED_RESULTS_PACKAGED',bundle,bundle.stat().st_size,digest,flush=True)
if __name__=='__main__':main()
