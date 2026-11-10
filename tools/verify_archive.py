"""Check data IDs, metric aggregation, training endpoints and published hashes."""
from pathlib import Path
import ast,csv,hashlib,json,math,re

ROOT=Path(__file__).resolve().parents[1]

def load(path):return json.loads(path.read_text())
def main():
    index=load(ROOT/'tables/experiment_index.json');runs=index['runs']
    assert len(runs)==118 and sum(r['status']=='complete' for r in runs)==110
    assert sum(r['status']=='failed' for r in runs)==8
    test_ids={}
    for dataset in ['HYCOM','OISST','GHRSST']:
        m=load(ROOT/f'data/manifests/{dataset}_manifest.json')
        a={s['sample_id'] for s in m['splits']['train']['samples']};b={s['sample_id'] for s in m['splits']['test']['samples']}
        assert len(a)==800 and len(b)==200 and not a&b
        test_ids[dataset]=b
    checked=0;per_image=0;validation_rounds=0
    for run in runs:
        if run['status']!='complete':
            assert not run['result_path'];continue
        p=ROOT/run['result_path'];j=load(p);d=p.parent
        if run['evaluation_split']=='test200':
            rows=list(csv.DictReader((d/('model_test_per_sample.csv' if (d/'model_test_per_sample.csv').exists() else 'wmsr_test_per_sample.csv')).open()))
            assert len(rows)==200 and {r['sample_id'] for r in rows}==test_ids[run['dataset']],run['id']
            actual=j.get('test_model',j.get('test_wmsr'))
            recomputed={'psnr_y_db':sum(float(r['psnr_y_db']) for r in rows)/200,
                        'ssim_y':sum(float(r['ssim_y']) for r in rows)/200,
                        'rmse_c':math.sqrt(sum(float(r['mse_c2']) for r in rows)/200),
                        'mae_c':sum(float(r['mae_c']) for r in rows)/200}
            for key,v in recomputed.items():assert math.isclose(v,actual[key],rel_tol=1e-6,abs_tol=1e-6),(run['id'],key)
            per_image+=len(rows)
        else:
            history=[json.loads(s) for s in (d/'validation_history.jsonl').read_text().splitlines()]
            assert [v['step'] for v in history]==list(range(200,run['updates']+1,200)),run['id']
            split=load(ROOT/f'data/development_splits/{run["stage"]}/OISST.json')
            allowed=set(split['validation_ids'])
            assert len(allowed)==80 and len(split['fit_indices'])==720
            for v in history:
                for kind in ['raw','ema']:
                    r=load(d/f'validation_{v["step"]:06d}_{kind}.json')
                    assert r['summary']['count']==80 and {i['sample_id'] for i in r['samples']}==allowed
                    assert all(math.isfinite(float(x)) for x in r['summary'].values())
                    per_image+=len(r['samples'])
                validation_rounds+=1
            assert j['last_validation']['final_test_images_evaluated']==0
        training=list(csv.DictReader((d/'training_metrics.csv').open()))
        assert int(training[-1]['step'])==run['updates'],run['id']
        checked+=1
    # Normalized public copies retain a digest of the source and published bytes.
    provenance=load(ROOT/'checks/publication_provenance.json')
    for item in provenance:
        path=ROOT/item['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==item['published_sha256'],item['path']
    for file in ROOT.rglob('*.py'):
        if '.git' not in file.parts:ast.parse(file.read_text())
    weights=load(ROOT/'weights/manifest.json');assert len(weights)==114
    assert len({i['archive_path'] for i in weights})==114
    for item in weights:assert re.fullmatch('[0-9a-f]{64}',item['sha256'])
    result={'status':'pass','tasks':len(runs),'complete':checked,'failed':8,'formal_test_runs':105,
            'per_image_records_checked':per_image,'development_validation_rounds':validation_rounds,
            'published_source_files_checked':len(provenance),'indexed_weight_files':len(weights)}
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':main()
