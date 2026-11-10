#!/usr/bin/env python3
"""Verify one candidate screen and compare with the frozen batch1 WMSR reference."""
import csv
import datetime
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    root=HERE/'cloud_recovered/pilot/results/ESC_OISST_x4_s10'
    retrieval=json.loads((HERE/'records/retrieval_status.json').read_text())
    assert retrieval['complete_candidate_pilot'] and retrieval['all_file_hashes_match']
    assert len(retrieval['pilot_weights'])==1
    result=json.loads((root/'pilot_results.json').read_text())
    config=json.loads((root/'config.json').read_text())
    protocol=json.loads((HERE/'pilot_protocol.json').read_text())
    split=json.loads((HERE/'development_splits/OISST.json').read_text())
    history=[json.loads(x) for x in (root/'validation_history.jsonl').read_text().splitlines()]
    assert result['updates']==1000 and [r['step'] for r in history]==[200,400,600,800,1000]
    assert config['protocol']==protocol and config['final_test_images_evaluated']==0
    training=list(csv.DictReader((root/'training_metrics.csv').open()))
    assert [int(r['step']) for r in training]==list(range(100,1001,100))
    for row in history:
        assert row['final_test_images_evaluated']==0
        for kind in ['raw','ema']:
            d=json.loads((root/f'validation_{row["step"]:06d}_{kind}.json').read_text())
            assert d['summary']==row[kind] and len(d['samples'])==d['summary']['count']==80
            assert [s['sample_id'] for s in d['samples']]==split['validation_ids']
    assert sha(root/'checkpoints/pilot_final.pt')==result['checkpoint_sha256']==retrieval['pilot_weights'][0]['sha256']
    content=json.loads((HERE/'cloud_recovered/records/final_checkpoint_verification.json').read_text())
    assert content['step']==content['optimizer_updates']==1000 and content['checkpoint_sha256']==result['checkpoint_sha256']
    assert all(content['WMSR_reference_same_training_order_and_augmentation'].values())
    reference=json.loads((HERE/'cloud_recovered/records/esc_profile_and_reference.json').read_text())
    assert reference['status']=='pass' and reference['reference_recheck']['old_results_modified']==False
    baseline=reference['reference_recheck']['metrics'];current=history[-1]['raw']
    bicubic_record=json.loads((HERE/'cloud_recovered/records/bicubic_development80.json').read_text())
    bicubic=bicubic_record['metrics']
    delta={k:current[k]-baseline[k] for k in ['psnr_y_db','rmse_c','mae_c','ssim_y']}
    verdict='本轮1000步末点训练内验证胜出' if delta['psnr_y_db']>0 and delta['rmse_c']<0 else '本轮1000步末点未同时胜出PSNR与RMSE'
    verification={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'pass','updates':1000,'validation_rounds':5,'validation_count_each':80,'final_test200_evaluated':False,'weight_sha256':result['checkpoint_sha256'],'all_validation_ids_match':True,'equal_common_control_fields':reference['equal_control_fields'],'baseline_checkpoint_sha256':reference['reference_recheck']['checkpoint_sha256'],'baseline_batch1_recheck_abs_differences':reference['reference_recheck']['absolute_difference_from_original'],'new_long_training_started':False,'resume_and_data_exposure_checks':content}
    (HERE/'records/screen_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    analysis={'verification':verification,'ESC_raw_1000':current,'WMSR_raw_1000_same80_batch1':baseline,'bicubic_same80':bicubic,'ESC_minus_WMSR':delta,'verdict':verdict,'history':history,'ESC_ema_1000':history[-1]['ema'],'boundary':'Training-only80 validation at1000updates/one dataset/one scale/one seed; not original200 test or original-paper superiority; no convergence claim.'}
    (HERE/'records/screen_comparison.json').write_text(json.dumps(analysis,indent=2,ensure_ascii=False)+'\n')
    completed_local=datetime.datetime.fromisoformat(result['completed_at']).astimezone(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M')
    text=['# 再筛选结论与ESC短程预实验','',f'完成于{completed_local}（北京时间）：候选选为ESC标准版（ICCV2025），并完成一组OISST×4、1000步的训练内验证预实验。**{verdict}。**','',
          '## 与WMSR的当前实测','',
          '| 同一80图验证集，1000步末点raw参数 | PSNR↑/dB | RMSE↓/℃ | SSIM↑ |','|---|---:|---:|---:|',
          f"| ESC标准版 | {current['psnr_y_db']:.4f} | {current['rmse_c']:.6f} | {current['ssim_y']:.6f} |",
          f"| WMSR公开4-WAM网络 | {baseline['psnr_y_db']:.4f} | {baseline['rmse_c']:.6f} | {baseline['ssim_y']:.6f} |",'',
          f"同80图的无训练Bicubic基线为PSNR {bicubic['psnr_y_db']:.4f}dB、RMSE {bicubic['rmse_c']:.6f}℃。WMSR1000步的PSNR略低于插值、RMSE略好；逐图PSNR均值与整体温度RMSE的排序不必一致。这提醒我们先检查充分收敛和插值增益，不能把网络年代或参数量当成任务优势。",'',
          f"ESC减WMSR：PSNR {delta['psnr_y_db']:+.4f}dB，RMSE {delta['rmse_c']:+.6f}℃。WMSR使用原已完成1000步权重，另以batch1在同80图重评，PSNR与原batch2结果相差约{reference['reference_recheck']['absolute_difference_from_original']['psnr_y_db']:.2e}dB；没有改旧成绩或重复训练它。",'',
          '**这不是原200张测试集成绩，不是论文表格值，也不代表已经完成九组正式训练或证明优于充分训练的WMSR。**80图由原训练800中按日期冻结划出，其余720张用于拟合；原200测试图本轮未参与模型选择。单seed、单数据集、单倍率、1000步只能作为候选筛选证据。','',
          '## ESC学习曲线','',
          '| 更新步 | raw PSNR | raw RMSE/℃ | EMA PSNR |','|---:|---:|---:|---:|']
    for r in history:text.append(f"| {r['step']} | {r['raw']['psnr_y_db']:.4f} | {r['raw']['rmse_c']:.6f} | {r['ema']['psnr_y_db']:.4f} |")
    text+=['','![ESC训练内验证学习曲线](./cloud_recovered/pilot/results/screen_learning_curves.png)','',
           '本实验在832步按用户要求暂停，2026-11-10核对完整断点与源码哈希后续跑168步至1000。最终采样顺序、位置、采样RNG与增强RNG均与既有WMSR的1000步记录逐项一致，优化器更新计数核对为1000。CSV中900步这一行的mean_l1_last_window仅统计恢复后的68步833～900，因为暂停前832步的局部损失累计不在断点中；这影响该行统计窗口，不影响模型更新、EMA或样本曝光次数。',]
    text+=['','EMA从初始随机权重开始累计、decay=.999，在1000步时仍显著滞后，因此保留其诊断值，以raw末点作本轮短程比较。没有依据原200测试集挑选checkpoint，也没有只展示最有利的中间点。','',
           '## 选型依据与尚存边界','',
           'ESC由University of Seoul的Dongheon Lee、Seokju Yun、Youngmin Ro提出，与WMSR作者不同。选标准版，三倍率946728/955383/967500参数均在5090实例化核对；不是ESC-light或ESC-Real。其卷积与注意力混合结构、原生图像跳连提供了不同于PFT的候选方向，但不能单凭这些结构描述保证SST泛化。自然图像论文中的高分只作选型参考。[论文](https://arxiv.org/html/2503.06671v2)、[官方源码](https://github.com/dslisleedh/ESC)','',
           '固定官方commit：`f22c712a57c664360f9af3a77cfc527c414067c4`。采用原生Naive FP32注意力与完整相对位置偏置，跳过BasicSR注册器依赖。原生标准版eval分支要求batch1，本轮已遵守；将未注册为buffer的派生整数相对位置索引放到GPU缓存，避免每次CPU传输。直接batch8/CPU索引与micro2累计4/GPU索引的输出、损失及参数梯度已通过数值比较。没有改其卷积/注意力方程、层数、通道、初始化或预训练权重。','',
           '两模型共同控制项已逐项核对：相同数据划分/数量/ID、seed10、1000次更新、有效batch8、micro2累计4、AdamW2e-4/betas(.9,.99)/weight_decay0、纯L1/0～1、EMA.999、100步预热后恒定LR、增强和FP32精度。不同网络保留各自原生初始化；这不是相同初始权重分布的实验。ESC原自然图像配方为更长训练及不同优化设置，本轮并非该配方的完整复现，不能把短预算结果当作原论文能力上限。','',
           'PFT额外复核在8张训练图上的train/eval输出差为0，未发现模式切换错误；原负结果保留。其验证波动仍不能归因于某一已证明原因。','',
           '## 新近备选与下一步','',
           '另检查了UCAN（CVPR2026，作者不同），但固定仓库缺少README指向的训练配置及requirements.txt，未见独立UCAN-L配置，不能把公开配置随意称为大模型。UADM（TGRS2026）专门做SST，但官方为OISST→OSTIA×5并有硬编码5倍结构、扩散/判别器及地理元数据，不能不经说明塞进原三倍率单输入协议。DBSD-Net与WMSR作者团队重合，已排除。详见[逐项筛选记录](./candidate_audit.json)。','',
           f"×4资源探针中，ESC每有效batch更新中位数约{reference['profile']['median_seconds_per_update']:.3f}秒，带AdamW和EMA的峰值已分配显存约{reference['profile']['peak_allocated_mib_with_optimizer_and_ema']/1024:.2f}GiB；这是3次预热后5次测量，不等于九组全程成本。此次实际1000步含验证与断点耗时约{result['elapsed_seconds']/60:.1f}分钟。",'',
           '未自动扩展为长训练。若继续，建议先在新版本协议与新目录将ESC、WMSR各从本次1000步完整断点（含优化器、EMA和采样RNG）延续到5000步，用同80图观察收敛；不回改本轮1000步记录。按当前×4两网络的实测速率，新增各4000步纯训练约48分钟，含验证和存档约50～60分钟，实际仍可能变化。正式阶段再使用完整800训练/200测试及原三个数据集、三个倍率，不减少数据、不换测试集，不引入额外自然图像训练。固定测试集已有历史多轮观察，后续比较仍属于探索，不应称为从未使用的独立确认。','',
           '## 文件与校验','',
           '- [候选配置和源码版本](./selected_candidate.json)、[源文件哈希](./source_manifest.json)。',
           '- [全部验收](./records/screen_verification.json)、[三倍率GPU预检](./records/esc_preflight.json)。',
           '- [逐图验证、配置、CSV与最终权重回收证明](./records/retrieval_status.json)。',
           '- [ESC 1000步raw+EMA权重](./cloud_recovered/pilot/results/ESC_OISST_x4_s10/checkpoints/pilot_final.pt)。',
           '- [完整数值](./records/screen_comparison.json)、[WMSR参考重评与资源探针](./records/esc_profile_and_reference.json)。','',
           '云端目录为`/root/wmsr-repro/strong_models/candidate_screen_20261109`，新增文件与缓存在新目录；旧86组、v2-Light九组与PFT两组结果均未改动。本地保留重要论文/源码/协议/日志/结果/关键权重，没有安装本地训练环境。短程实验完成后无自动长训练，云平台停机及计费状态仍需用户确认。','']
    (HERE/'筛选结论与ESC预实验.md').write_text('\n'.join(text))
    print(json.dumps({'verification':'pass','verdict':verdict,'ESC':current,'WMSR':baseline,'delta':delta,'report':str(HERE/'筛选结论与ESC预实验.md')},indent=2,ensure_ascii=False))


if __name__=='__main__':main()
