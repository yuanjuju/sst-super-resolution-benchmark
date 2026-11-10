#!/usr/bin/env python3
"""Verify completed screen/resume metadata and render training-only curves."""
from pathlib import Path
import datetime
import hashlib
import json
import os

HERE=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(HERE/'.cache/matplotlib')
import torch


def main():
    run=HERE/'pilot/results/ESC_OISST_x4_s10'
    result=json.loads((run/'pilot_results.json').read_text())
    progress=json.loads((run/'progress.json').read_text())
    assert progress['status']=='complete' and progress['step']==result['updates']==1000
    final=run/'checkpoints/pilot_final.pt'
    assert hashlib.sha256(final.read_bytes()).hexdigest()==result['checkpoint_sha256']
    c=torch.load(final,map_location='cpu',weights_only=True)
    assert c['step']==1000 and c['config']==json.loads((run/'config.json').read_text())
    assert c['model'].keys()==c['ema'].keys()
    assert all(torch.isfinite(v).all() for d in ['model','ema'] for v in c[d].values())
    del c
    state=torch.load(run/'checkpoints/last.pt',map_location='cpu',weights_only=False)
    baseline=Path(state['config']['protocol']['baseline_reference'])
    reference=torch.load(baseline/'checkpoints/last.pt',map_location='cpu',weights_only=False)
    rng_equal={'sampler_rng':bool(torch.equal(state['rng_loader'],reference['rng_loader'])),
               'sample_order':state['order']==reference['order'],
               'sample_position':state['position']==reference['position'],
               'augmentation_rng':state['rng_augmentation']==reference['rng_augmentation']}
    assert all(rng_equal.values()),rng_equal
    assert all(float(v['step'])==1000 for v in state['optimizer']['state'].values() if 'step' in v)
    proof={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'step':1000,
           'checkpoint_sha256':result['checkpoint_sha256'],'checkpoint_content_and_config_verified':True,
           'all_model_and_ema_tensors_finite':True,'optimizer_updates':1000,
           'WMSR_reference_same_training_order_and_augmentation':rng_equal,
           'resumed_from_step':832,'new_updates_after_resume':168,
           'CSV_boundary_note':'First post-resume mean_l1_last_window is the68 updates833..900; partial pre-pause loss totals were not checkpointed. Model/optimizer/EMA/sampler/augmentation states and update count were preserved.',
           'test200_evaluated':False,'long_training_started':False}
    (HERE/'records/final_checkpoint_verification.json').write_text(json.dumps(proof,indent=2)+'\n')
    del state,reference
    history=[json.loads(s) for s in (run/'validation_history.jsonl').read_text().splitlines()]
    baseline=json.loads((HERE/'records/esc_profile_and_reference.json').read_text())['reference_recheck']['metrics']
    bicubic=json.loads((HERE/'records/bicubic_development80.json').read_text())['metrics']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,key,label in [(axs[0],'psnr_y_db','PSNR (dB, higher is better)'),(axs[1],'rmse_c','RMSE (C, lower is better)')]:
        ax.plot([r['step'] for r in history],[r['raw'][key] for r in history],marker='o',color='#c55a11',label='ESC raw')
        ax.axhline(baseline[key],color='#1768ac',linestyle='--',label='WMSR raw @1000')
        ax.axhline(bicubic[key],color='#357a38',linestyle=':',label='Bicubic')
        ax.set(xlabel='Optimizer updates',ylabel=label);ax.grid(alpha=.25);ax.legend()
    fig.suptitle('OISST x4 | same training-only development80\nESC seed10 pilot, not final200 test scores')
    for ext in ['png','svg']:fig.savefig(HERE/f'pilot/results/screen_learning_curves.{ext}',dpi=160)
    plt.close(fig)
    print(json.dumps(proof),flush=True)


if __name__=='__main__':main()
