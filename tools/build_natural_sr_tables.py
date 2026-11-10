"""Build the natural-image SR index from the archived measurements."""
from pathlib import Path
import csv
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'foundations/natural-image-sr'
OUT = ROOT / 'tables/natural_image_sr'


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ledger = read_csv(MODULE / 'results/metrics.csv')
    write_csv('metrics.csv', ledger)
    paired = read_csv(MODULE / 'results/safmn_fft_pair_30k/results/training_metrics.csv')
    training = []
    for row in paired:
        training.append(dict(experiment='SAFMN_pair_5090_' + row['arm'],
            model='SAFMN', step=int(row['iter']), kind='scheduled',
            psnr_rgb_db=float(row['psnr_rgb_db']), weight_key='params_ema',
            dataset='DIV2K_val100', crop_border=4,
            source='foundations/natural-image-sr/results/safmn_fft_pair_30k/' + row['arm'] + '.stdout.log'))
    cam = json.loads((MODULE / 'results/camixersr5000.result.json').read_text())
    for row in cam['validation']:
        training.append(dict(experiment='CAMixerSR_5090_5k',model='CAMixerSR',
            step=row['step'],kind=row['kind'],psnr_rgb_db=row['psnr_db'],
            weight_key='params_ema',dataset='DIV2K_val100',crop_border=4,
            source='foundations/natural-image-sr/results/logs/camixersr5000.stdout.log'))
    for row in ledger:
        if row['model'] == 'SAFMN' and row['notes'].startswith('training validation'):
            step = int(re.search(r'scratch_(\d+)',row['checkpoint']).group(1))
            nofft = 'nofft' in row['checkpoint']
            log = 'nofft5000.stdout.log' if nofft else ('first5000.stdout.log' if step==5000 else 'full500k_resume.stdout.log')
            training.append(dict(experiment='SAFMN_4090_' + ('nofft' if nofft else 'fft'),
                model='SAFMN',step=step,kind='scheduled',psnr_rgb_db=float(row['psnr_db']),
                weight_key='params_ema',dataset='DIV2K_val100',crop_border=4,
                source='foundations/natural-image-sr/results/logs/' + log))
    write_csv('training_validation.csv', training)
    set5 = read_csv(MODULE / 'results/safmn_fft_pair_30k/results/set5_per_image.csv')
    for row in set5:
        row['prediction_path'] = 'foundations/natural-image-sr/results/safmn_fft_pair_30k/results/predictions/' + row['image'] + '/' + row['arm'] + '_' + row['iter'] + '.png'
    write_csv('set5_per_image.csv',set5)
    means = read_csv(MODULE / 'results/safmn_fft_pair_30k/results/set5_mean.csv')
    write_csv('set5_mean.csv',means)
    local = []
    for path in sorted((MODULE/'results/local-inference').glob('*/result.json')):
        row=json.loads(path.read_text())
        local.append(dict(run=path.parent.name,weight_key=row['weight_key'],
            psnr_rgb_db=row['metrics']['sr']['psnr_db'],
            pillow_bicubic_psnr_rgb_db=row['metrics']['pillow_bicubic']['psnr_db'],
            crop_border=4,device=row['device'],
            wall_seconds=row['inference_wall_seconds'],
            source=str(path.relative_to(ROOT))))
    write_csv('local_inference.csv',local)
    files=[]
    for path in sorted((MODULE/'results').rglob('*.log')):
        text=path.read_text()
        scores=re.findall(r'# psnr:\s*([0-9.]+)',text)
        files.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,
            psnr_events=len(scores),last_psnr_db=scores[-1] if scores else '',
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    write_csv('test_logs.csv',files)
    datasets=json.loads((MODULE/'data/manifest.json').read_text())
    write_csv('set5_files.csv',datasets)
    payload=dict(metrics=ledger,training=training,set5=set5,set5_mean=means,
        local=local,logs=files,datasets=datasets)
    (OUT/'workbook_source.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(metrics=len(ledger),validation_records=len(training),
        set5_predictions=len(set5),local_inference=len(local),logs=len(files),
        set5_files=len(datasets))))


if __name__=='__main__':
    main()
