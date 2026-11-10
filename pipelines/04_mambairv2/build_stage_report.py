#!/usr/bin/env python3
"""Summarize one fully recovered scale without filling unfinished results."""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parents[1] / '云端实验结果/论文补充实验/results/extended'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, choices=[2, 3, 4], required=True)
    scale = parser.parse_args().scale
    plan = json.loads((HERE / 'protocol.json').read_text())
    retrieval = json.loads((HERE / 'records/retrieval_status.json').read_text())
    verified = {r['id']: r for r in retrieval['verified']}
    jobs = [j for j in plan['jobs'] if j['scale'] == scale]
    assert len(jobs) == 3
    overall, proof = [], []
    for job in jobs:
        ident = job['id']
        record = verified[ident]
        folder = HERE / 'results' / ident
        assert all(digest(folder / rel) == meta['sha256'] for rel, meta in record['files'].items())
        new = json.loads((folder / 'final_results.json').read_text())
        config = json.loads((folder / 'config.json').read_text())
        ref_folder = OLD / job['reference_id']
        old = json.loads((ref_folder / 'final_results.json').read_text())
        ref = json.loads((ref_folder / 'config.json').read_text())
        assert config == new['config']
        assert digest(ref_folder / 'config.json') == job['reference_config_sha256']
        assert all(config[k] == ref[k] == value for k, value in job['controlled_reference'].items())
        assert config['data_manifest_sha256'] == ref['data_manifest_sha256'] == job['data_manifest_sha256']
        assert new['epochs'] == 100 and new['optimizer_steps'] == 10000
        assert new['test_model']['count'] == old['test_model']['count'] == 200
        assert new['test_bicubic'] == old['test_bicubic']
        overall.append({'dataset': job['dataset'], 'scale': scale,
                        'WMSR': old['test_model'], 'MambaIRv2Light': new['test_model'],
                        'delta_psnr_db': new['test_model']['psnr_y_db'] - old['test_model']['psnr_y_db'],
                        'bicubic': new['test_bicubic'], 'parameters': new['parameters']})
        proof.append({'id': ident, 'reference_id': job['reference_id'],
                      'reference_config_sha256': job['reference_config_sha256'],
                      'checkpoint_sha256': record['checkpoint_sha256'],
                      'public_controls_match': True, 'data_manifest_match': True,
                      'config_matches_result': True, 'bicubic_metrics_match': True})
    with (HERE / '三方对照表_待新模型实测.csv').open(newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if int(r['scale']) == scale]
    assert len(rows) == 6
    assert all(r['MambaIRv2Light_status'] == 'TRAINED_TESTED_VERIFIED' for r in rows)
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    lines = [f'# ×{scale}阶段：WMSR原文、既有复现与MambaIRv2-Light实测', '',
             f'生成时间：{now}。本报告仅覆盖×{scale}三套SST数据，整轮进度见[当前状态](./当前状态.md)。', '',
             '三组均完成100epochs/10000updates及200张固定测试图评测，最终权重、配置、训练CSV及逐图指标已回收。逐文件SHA256、100条训练记录、200个唯一测试样本及公共控制字段均已核验；旧参考配置哈希、配置与结果内嵌配置、Bicubic指标也一致。', '',
             '## 同协议实测结果', '',
             '| 数据集 | 既有WMSR PSNR(Y) | MambaIRv2-Light PSNR(Y) | 新模型−WMSR / dB | Bicubic PSNR(Y) |',
             '| --- | --- | --- | --- | --- |']
    for r in overall:
        lines.append(f"| {r['dataset']} | {r['WMSR']['psnr_y_db']:.4f} | {r['MambaIRv2Light']['psnr_y_db']:.4f} | {r['delta_psnr_db']:+.4f} | {r['bicubic']['psnr_y_db']:.4f} |")
    below = sum(r['delta_psnr_db'] < 0 for r in overall)
    lines += ['', f'在固定SST数据、训练预算与配方下，本倍率新模型的PSNR在{below}/3套数据上低于既有WMSR。保留全部实测结果，不修改冻结配置追分。', '',
              '这不是MambaIRv2原论文自然图像训练配方的复现，也不代表其充分调优后的能力上限；单种子结果不能证明普遍优劣。两网络原生初始化、内部归一化、扫描/填充规则和参数数量按原定义保留。', '',
              '## 原文参考列与两列实测', '',
              '**原文列是WMSR论文Table III–V的作者报告值，只作文献参考。**作者原样本及完整出表实现未恢复，不能将两列实测的高绝对分数解释为超越原文。HYCOM使用原文四区域对应结果，不混用整体均值。', '',
              '| 数据集/区域 | 原文WMSR PSNR / SSIM | 既有实测WMSR PSNR(Y) / SSIM(Y) | 实测MambaIRv2-Light PSNR(Y) / SSIM(Y) |',
              '| --- | --- | --- | --- |']
    for r in rows:
        label = r['dataset'] + (f" / {r['region']}" if r['region'] else '')
        lines.append(f"| {label} | {float(r['paper_psnr']):.2f} / {float(r['paper_ssim']):.4f} | {float(r['current_psnr_y_db']):.4f} / {float(r['current_ssim_y']):.6f} | {float(r['MambaIRv2Light_psnr_y_db']):.4f} / {float(r['MambaIRv2Light_ssim_y']):.6f} |")
    lines += ['', '## 温度误差', '',
              '| 数据集 | WMSR RMSE/°C | 新模型 RMSE/°C | WMSR MAE/°C | 新模型 MAE/°C |',
              '| --- | --- | --- | --- | --- |']
    for r in overall:
        lines.append(f"| {r['dataset']} | {r['WMSR']['rmse_c']:.6f} | {r['MambaIRv2Light']['rmse_c']:.6f} | {r['WMSR']['mae_c']:.6f} | {r['MambaIRv2Light']['mae_c']:.6f} |")
    lines += ['', '物理误差按固定[-2,40]°C映射反变换并采用相同裁边，不等于作者未公开的温度编码协议。', '',
              '## 证据与后续', '',
              '- [三方完整表（未完成点留空）](./三方对照表_待新模型实测.csv)',
              '- [冻结协议及模型来源](./README.md)',
              '- [逐文件回收证明](./records/retrieval_status.json)',
              f'- [本倍率对照与额外核验](./records/scale{scale}_comparison.json)', '',
              '继续按九组既定清单运行其余任务；全部训练、评测与回收校验完成后生成最终报告。']
    output = HERE / f'×{scale}阶段对照.md'
    output.write_text('\n'.join(lines) + '\n')
    record = {'generated_at': now, 'scale': scale, 'overall': overall, 'verification': proof,
              'inputs': {name: digest(HERE / name) for name in ['protocol.json', 'records/retrieval_status.json', '三方对照表_待新模型实测.csv']},
              'report_sha256': digest(output)}
    (HERE / f'records/scale{scale}_comparison.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'report': str(output), 'scale': scale, 'verified_jobs': len(overall),
                      'deltas_db': {r['dataset']: r['delta_psnr_db'] for r in overall}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
