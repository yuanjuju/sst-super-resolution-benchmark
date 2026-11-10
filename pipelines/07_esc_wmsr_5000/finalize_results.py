"""Verify prescribed endpoint and every validation ID, then write factual report."""
import csv
import datetime
import json
import math
from pathlib import Path
import torch
from screen_helpers import HERE, sha


def main():
    protocol = json.loads((HERE / 'protocol.json').read_text())
    split = json.loads((HERE / 'development_splits/OISST.json').read_text())
    queue = json.loads((HERE / 'records/queue_status.json').read_text())
    assert queue['status'] == 'complete' and queue['completed'] == ['ESC', 'WMSR']
    expected = list(range(200, 5001, 200))
    rows, states, proof = [], {}, {}
    for name in ['ESC', 'WMSR']:
        run = HERE / 'pilot/results' / f'{name}_OISST_x4_s10'
        config = json.loads((run / 'config.json').read_text())
        assert config['protocol'] == protocol
        assert config['protocol_sha256'] == sha(HERE / 'protocol.json')
        assert config['trainer_sha256'] == sha(HERE / 'train_continue.py')
        assert config['helper_sha256'] == sha(HERE / 'screen_helpers.py')
        assert config['model_adapter_sha256'] == sha(HERE / 'model_adapter.py')
        assert config['final_test_images_evaluated'] == 0
        result = json.loads((run / 'pilot_results.json').read_text())
        assert result['status'] == 'complete' and result['updates'] == 5000
        history = [json.loads(s) for s in (run / 'validation_history.jsonl').read_text().splitlines()]
        assert [r['step'] for r in history] == expected, name
        with (run / 'training_metrics.csv').open() as f: training = list(csv.DictReader(f))
        assert [int(r['step']) for r in training] == list(range(100, 5001, 100))
        assert all(float(r['lr']) == .0002 for r in training)
        for validation in history:
            assert validation['final_test_images_evaluated'] == 0
            for variant in ['raw', 'ema']:
                data = json.loads((run / f'validation_{validation["step"]:06d}_{variant}.json').read_text())
                assert data['summary']['count'] == 80
                assert [r['sample_id'] for r in data['samples']] == split['validation_ids']
                aggregate = {'count':80,
                             'rmse_c':math.sqrt(sum(r['mse_c2'] for r in data['samples'])/80),
                             'mae_c':sum(r['mae_c'] for r in data['samples'])/80,
                             'psnr_y_db':sum(r['psnr_y_db'] for r in data['samples'])/80,
                             'ssim_y':sum(r['ssim_y'] for r in data['samples'])/80}
                for key,value in aggregate.items():
                    assert math.isclose(value, data['summary'][key], rel_tol=1e-12, abs_tol=1e-12)
                    assert math.isclose(value, validation[variant][key], rel_tol=1e-12, abs_tol=1e-12)
        path = run / 'checkpoints/pilot_final.pt'
        assert sha(path) == result['checkpoint_sha256']
        compact = torch.load(path, map_location='cpu', weights_only=True)
        full_path = run / 'checkpoints/last.pt'
        full = torch.load(full_path, map_location='cpu', weights_only=False)
        assert compact['step'] == full['step'] == 5000
        assert compact['config'] == full['config'] == config
        for kind in ['model','ema']:
            assert compact[kind].keys() == full[kind].keys()
            assert all(torch.isfinite(t).all() and torch.equal(t,full[kind][k]) for k,t in compact[kind].items())
        optimizer_steps = sorted(set(float(s['step']) for s in full['optimizer']['state'].values() if 'step' in s))
        assert optimizer_steps == [5000.]
        assert sha(HERE / config['resume_lineage']['local_input']) == config['resume_lineage']['source_checkpoint_sha256']
        # Sources are read only throughout continuation, including their original checkpoint bytes.
        assert sha(Path(config['resume_lineage']['source_checkpoint'])) == config['resume_lineage']['source_checkpoint_sha256']
        states[name] = full
        proof[name] = {'step':5000,'validation_rounds':25,'new_validation_rounds':20,'each_count':80,
                       'optimizer_steps':optimizer_steps,'compact_sha256':sha(path),'full_resume_sha256':sha(full_path),
                       'final_raw':history[-1]['raw'],'final_ema':history[-1]['ema'], 'source1000':history[4],
                       'all_validation_ids_verified':True,'all_model_and_ema_tensors_finite':True}
        for h in history:
            for variant in ['raw','ema']:
                rows.append({'model':name,'step':h['step'],'variant':variant,**h[variant]})
    a,b = states['ESC'],states['WMSR']
    assert a['order'] == b['order'] and a['position'] == b['position']
    assert a['rng_augmentation'] == b['rng_augmentation'] and torch.equal(a['rng_loader'],b['rng_loader'])
    report = {'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'pass','models':proof,
              'same_final_sampler_and_augmentation':True,'sampling_position':a['position'],
              'test200_evaluated':False,'equal_common_protocol':True,'raw_delta_ESC_minus_WMSR':{
                  k:proof['ESC']['final_raw'][k]-proof['WMSR']['final_raw'][k] for k in ['psnr_y_db','rmse_c','mae_c','ssim_y']}}
    (HERE / 'records/final_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    with (HERE / 'validation_learning_curves.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text=['# ESC与WMSR：5000步成对续训结果','',
          '本轮在同一OISST×4训练内划分上，将两模型由1000步续到5000步；恢复优化器、EMA、采样器和全部随机状态。720张拟合、80张验证，原200张测试集未参与。',
          '', '| 模型 / 参数 | 1000步 PSNR(dB) | 5000步 PSNR(dB) | 1000步 RMSE(℃) | 5000步 RMSE(℃) |',
          '|---|---:|---:|---:|---:|']
    for name in ['ESC','WMSR']:
        for variant in ['raw','ema']:
            before=proof[name]['source1000'][variant];after=proof[name]['final_'+variant]
            text.append(f'| {name} / {variant} | {before["psnr_y_db"]:.4f} | {after["psnr_y_db"]:.4f} | {before["rmse_c"]:.5f} | {after["rmse_c"]:.5f} |')
    delta=report['raw_delta_ESC_minus_WMSR']
    text.extend(['',f'固定5000步raw末点，ESC相对WMSR：PSNR差{delta["psnr_y_db"]:+.4f} dB，温度RMSE差{delta["rmse_c"]:+.5f} ℃。PSNR越高越好，RMSE越低越好，两者统计方式不同，应分别解释。',
                 '', '这是训练内开发实验，不能称作原论文200图测试结果，不能据此宣布普遍超越某篇论文。没有选取最好的中间步；raw与EMA均完整保留。1000步历史WMSR验证为batch2，独立batch1复核差约5.6e-8 dB；新增20轮验证两模型均为batch1。',
                 '', '公共控制：seed10、有效batch8=micro2×累计4、AdamW 2e-4 / betas(.9,.99) / weight_decay0、L1、EMA .999、前100步warmup后恒定LR、FP32且关闭两处TF32、每200步验证同一80图、每100步保存。网络各保留原生结构与初始化。',
                 '', '25轮验证（其中20轮为本次新增）、逐图ID、5000次优化器更新、最终权重和完整断点内容、两模型末点采样/增强状态已核验，详见records/final_verification.json。下载回本地后的SHA256证明由records/retrieval_status.json单独给出。',
                 '', '图表位于visualizations_20261110；固定沿用1000步可视化的前4个验证样本。正式测试和更长训练均需另行确定，本轮队列在5000步后结束。'])
    (HERE / '5000步对照结果.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({'status':'pass','final_raw':{n:proof[n]['final_raw'] for n in proof},'delta':delta}))


if __name__ == '__main__': main()
