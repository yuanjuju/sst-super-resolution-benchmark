#!/usr/bin/env python3
"""Verify recovered artifacts and summarize the frozen development pilot."""
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    recovered = HERE / 'cloud_recovered'
    protocol = json.loads((HERE / 'pilot_protocol.json').read_text())
    status = json.loads((HERE / 'records/live_status.json').read_text())
    retrieval = json.loads((HERE / 'records/retrieval_status.json').read_text())
    checkpoint_proof = json.loads((HERE / 'records/final_checkpoint_verification.json').read_text())
    split = json.loads((HERE / 'development_splits/OISST.json').read_text())
    assert status['queue']['status'] == 'complete' and not status['queue']['failed']
    assert retrieval['complete_paired_pilot'] and retrieval['all_file_hashes_match'] and len(retrieval['pilot_weights']) == 2
    configs, runs = {}, {}
    for model in protocol['models']:
        root = recovered / f'pilot/results/{model}_OISST_x4_s10'
        config = json.loads((root / 'config.json').read_text())
        assert config['protocol'] == protocol and config['final_test_images_evaluated'] == 0
        history = [json.loads(s) for s in (root / 'validation_history.jsonl').read_text().splitlines()]
        assert [r['step'] for r in history] == [200,400,600,800,1000]
        for row in history:
            assert row['final_test_images_evaluated'] == 0
            for kind in ['raw','ema']:
                d = json.loads((root / f'validation_{row["step"]:06d}_{kind}.json').read_text())
                assert d['summary'] == row[kind] and d['summary']['count'] == len(d['samples']) == 80
                assert [s['sample_id'] for s in d['samples']] == split['validation_ids']
                assert all(math.isfinite(float(s[k])) for s in d['samples'] for k in ['mse_c2','mae_c','psnr_y_db','ssim_y'])
        training = list(csv.DictReader((root / 'training_metrics.csv').open()))
        assert [int(r['step']) for r in training] == list(range(100,1001,100))
        result = json.loads((root / 'pilot_results.json').read_text())
        assert result['updates'] == 1000 and result['last_validation'] == history[-1]
        checkpoint = root / 'checkpoints/pilot_final.pt'
        assert sha(checkpoint) == result['checkpoint_sha256'] == checkpoint_proof['models'][model]['sha256']
        assert checkpoint_proof['models'][model]['step'] == 1000 and checkpoint_proof['models'][model]['protocol_matches']
        configs[model] = config
        runs[model] = {'result':result,'history':history,'training':training,'local_checkpoint':str(checkpoint),
                       'best_recorded_raw_psnr':max(history,key=lambda r:r['raw']['psnr_y_db']),
                       'minimum_recorded_raw_rmse':min(history,key=lambda r:r['raw']['rmse_c'])}
    differences = [k for k in configs['PFT'] if configs['PFT'][k] != configs['WMSR'][k]]
    assert differences == ['model'], differences
    profile = json.loads((recovered / 'records/model_profile.json').read_text())
    hours10k = sum(r['median_seconds_per_update'] for r in profile['batches']) * 3 * 10000 / 3600
    verification = {'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'pass',
                    'tasks_complete':2,'updates_per_task':1000,'validation_rounds_per_task':5,'samples_per_validation_round':80,
                    'validation_ids_match_frozen_manifest':True,'final_test_images_evaluated':0,
                    'common_config_fields_equal':True,'config_differences':differences,
                    'recovered_weights_sha256_match':{m:r['result']['checkpoint_sha256'] for m,r in runs.items()},
                    'long_training_started':False,'local_package_archive_sha256':retrieval['archive_sha256'],
                    'gpu_at_last_live_check':status['gpu'],'final_live_check_at':status['collected_at']}
    (HERE / 'records/pilot_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    summary = {'verification':verification,'runs':runs,'formal18job_training_hours':{'10000updates':hours10k,'20000updates':hours10k*2},
               'user_preference':'先看预实验结果，再决定长训练预算','recommendation':'Do not automatically expand to18jobs. Review PFT validation instability and EMA lag before choosing a bounded follow-up or another model.'}
    (HERE / 'records/pilot_analysis.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    p, w = runs['PFT']['result']['last_validation'], runs['WMSR']['result']['last_validation']
    lines = ['# PFT与WMSR成对预实验总结','',
             '2026-11-09：两个预实验均完成1000次更新，配置、每组5次80图验证、两份最终权重及回收SHA256全部核验通过。**本轮没有实现PFT超过WMSR：1000步末的raw参数验证PSNR为PFT 35.06 dB、WMSR 51.93 dB。暂不建议据此直接投入18组长训练。**','',
             '这些是原训练800张内部划出的80张验证图结果，**不是原200张测试成绩，也不是论文原文的测试结果**。单一OISST×4、单seed、1000步的结果不能代表模型充分收敛后的能力上限，更不能声称已完成三个数据集、九组倍率的正式对照。','',
             '## 同一步数的验证结果','',
             '| 1000步参数 | PFT PSNR↑ | WMSR PSNR↑ | PFT RMSE↓ | WMSR RMSE↓ |','|---|---:|---:|---:|---:|']
    for kind,label in [('raw','raw，当前网络参数'),('ema','EMA，指数平均参数')]:
        lines.append(f"| {label} | {p[kind]['psnr_y_db']:.4f} dB | {w[kind]['psnr_y_db']:.4f} dB | {p[kind]['rmse_c']:.4f} ℃ | {w[kind]['rmse_c']:.4f} ℃ |")
    lines += ['',f"raw参数下，PFT减WMSR的PSNR差为{p['raw']['psnr_y_db']-w['raw']['psnr_y_db']:+.4f} dB。WMSR在本轮短预算验证中明显更好。PSNR是逐图BT.601 Y/255/倍率边界裁剪后的均值；RMSE是裁剪后物理温度误差聚合，越低越好。",'',
              '## 学习曲线','',
              '| 更新步 | PFT raw PSNR | WMSR raw PSNR | PFT EMA PSNR | WMSR EMA PSNR |','|---:|---:|---:|---:|---:|']
    for a,b in zip(runs['PFT']['history'],runs['WMSR']['history']):
        lines.append(f"| {a['step']} | {a['raw']['psnr_y_db']:.4f} | {b['raw']['psnr_y_db']:.4f} | {a['ema']['psnr_y_db']:.4f} | {b['ema']['psnr_y_db']:.4f} |")
    lines += ['', '![验证学习曲线](./cloud_recovered/pilot/results/learning_curves.png)','',
              'PFT的训练L1总体下降，但验证PSNR在32.45～39.20 dB之间波动，末点35.06 dB；不能把训练损失下降当成验证精度已经稳定。WMSR验证PSNR从46.10提升到约52 dB，末点稍低于800步记录。两个网络在五次验证中的最高raw PSNR分别为PFT 39.1954 dB（400步）和WMSR 52.1807 dB（800步）；这是历史诊断记录，不代表这些中间权重已被选为正式checkpoint，预实验仅保留完整last与1000步紧凑权重。','',
              f'两者EMA从随机初始权重开始，decay=.999，1000步后参数平均中初始权重的系数仍约为{100*.999**1000:.2f}%。这能解释早期EMA滞后的一个机制，但不能定量解释全部误差，也不能解释为延长训练就一定让PFT反超。后续若评估EMA预热或不同优化方案，应先写入新协议，对两模型给予相同验证机会，不能回改本轮分数。','',
              '## 控制项与实现边界','',
              '两组共同采用原OISST训练800中的720张拟合、80张按完整日期分组验证，seed10，有效batch8=micro2累计4，AdamW lr2e-4/betas(.9,.99)/weight_decay0、纯L1、EMA.999、100步线性预热后恒定学习率、每200步验证，FP32并关闭两个TF32开关。所有公共配置字段逐项相同，config仅model字段不同；没有自然图像预训练，没有使用原200张测试图作选择。正式阶段若获同意，仍使用原800训练/200测试及三数据集三倍率，不能把720当成缩减后的正式训练集。','',
              'PFT保留固定官方完整网络拓扑、初始化和三倍率参数配置；其CUDA实现有明确修补：保留原文件，另存副本把两处grad_B并发非原子累加改为atomicAdd，并用动态导出名宏允许并排审计。原版在稠密参考反向校验失败，修补版在小尺寸和1024token/256topk输入通过。PFT×2直接batch8会OOM，因此双方共同采用梯度累积；×4与直接batch8的损失、输出和全部参数梯度校验已通过。这属于带记录的公共实现适配，不能写成未经修改的官方PFT数值复现。','',
              'WMSR是既有公开4-WAM网络，原生255范围经外部适配器映射为共同0～1损失范围。它与PFT结构、参数量和原生初始化不同，属于模型比较；此轮共同L1/AdamW/EMA训练路线与旧WMSR的L1+FFL配方也不同，不应把新验证分数直接与旧200图测试值排名。','',
              '原86组、MambaIRv2-Light九组、旧环境与原数据未改动。CUDA下载、头文件、扩展导出名、原版梯度、直接batch8 OOM、开启TF32时累积等价校验未通过等尝试均保留，没有隐藏负结果。队列日志另有一次完成后的幂等复核，跳过两份已完成结果，训练CSV与权重步数核验均为每组1000，没有追加更新。','',
              '## 正式预算与建议','',
              '| 假设预算：PFT九组 + WMSR九组 | 纯训练估计 |','|---|---:|',
              f'| 每组1万步 | {hours10k:.1f}小时 |',f'| 每组2万步 | {hours10k*2:.1f}小时 |','',
              '估计来自5090上各倍率3次预热后5次更新的中位数，包含AdamW和EMA，未包含完整验证、最终评测、文件回收、停机或故障时间，不能承诺收敛。实际本轮PFT约13.5分钟、WMSR约5.4分钟。','',
              '**建议暂缓直接跑18组。**先复核PFT验证波动与EMA启用方式，确认是否值得追加一个预先限定、双方同等机会的训练内试验，或另选适合小样本SST的候选。仅凭自然图像论文排名不能保证本SST协议中超过WMSR。用户已明确选择“先看预实验结果，再决定长训练预算”，目前未启动任何长训练，等待用户决定。','',
              '## 交付与云端状态','',
              '- [完整验收记录](./records/pilot_verification.json)：2组×1000步、每组5次80图验证、公共控制字段与权重哈希。',
              '- [逐文件回收证明](./records/retrieval_status.json)及[最终权重内容验收](./records/final_checkpoint_verification.json)。',
              '- [PFT 1000步权重（raw+EMA）](./cloud_recovered/pilot/results/PFT_OISST_x4_s10/checkpoints/pilot_final.pt)。',
              '- [WMSR 1000步权重（raw+EMA）](./cloud_recovered/pilot/results/WMSR_OISST_x4_s10/checkpoints/pilot_final.pt)。',
              '- [详细数值与完整曲线记录](./records/pilot_analysis.json)。','',
              f"最近云端核验时间：{status['collected_at']}；GPU：{status['gpu']}，空闲空间约{status['disk_free_gib']:.1f}GiB。两训练进程与队列均已退出，重要文件已本地回收校验。**可以在云平台关闭本轮实例；需由用户确认平台的停机状态和计费停止，SSH退出或GPU空闲不等于停止计费。**环境、原数据及可续训last.pt留在云盘。预实验总结与回收验收完成后删除本次巡检，不继续空跑。",'']
    report = HERE / '预实验总结.md'
    report.write_text('\n'.join(lines))
    print(json.dumps({'verification':'pass','report':str(report),'PSNR_raw_final':{m:r['result']['last_validation']['raw']['psnr_y_db'] for m,r in runs.items()},'verified_weights':verification['recovered_weights_sha256_match']},ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
