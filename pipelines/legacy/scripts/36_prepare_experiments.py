#!/usr/bin/env python3
"""Prespecified coverage of comparisons/TableVI/channel/group/loss experiments."""
import json,hashlib
from pathlib import Path
ROOT=Path('/root/wmsr-repro')
def ready_key(spec):
    architecture={k:spec[k] for k in ['model','scale','case','channels','groups','blocks'] if k in spec}
    return hashlib.sha256(json.dumps(architecture,sort_keys=True).encode()).hexdigest()[:16]
def main():
    jobs=[];aliases={}
    def add(identifier,kind,model,dataset,scale=4,epochs=100,**kwargs):
        spec=dict(id=identifier,kind=kind,model=model,dataset=dataset,scale=scale,epochs=epochs,seed=10,**kwargs)
        if model.startswith('WMSR'):spec.setdefault('l1_weight',.1);spec.setdefault('ffl_weight',.9)
        spec['preflight_key']=ready_key(spec);jobs.append(spec)
    for model in ['EDSR','ELAN','DAT','SAFMN','MambaIR','CRAN','WMSR']:
        for scale in [4,3,2]:
            for dataset in ['HYCOM','OISST','GHRSST']:add(f'compare_{model}_{dataset}_x{scale}_s10','comparison',model,dataset,scale)
    for case in range(1,10):add(f'ablation_case{case}_HYCOM_x4_s10','ablation','WMSR_ABLATION','HYCOM',epochs=20,case=case)
    for channels in [32,48,64,96,128]:
        identifier=f'channels_{channels}_GHRSST_x4_s10'
        if channels==64:aliases[identifier]='compare_WMSR_GHRSST_x4_s10'
        else:add(identifier,'channel_sweep','WMSR','GHRSST',channels=channels)
    for groups in [2,4,6]:
        for blocks in [2,4]:add(f'groups_{groups}_blocks_{blocks}_GHRSST_x4_s10','group_sweep','WMSR_GROUPED','GHRSST',groups=groups,blocks=blocks)
    for weight in [0.,.1,.5,.9,1.]:
        identifier=f'loss_l1_{weight}_GHRSST_x4_s10'
        if weight==.1:aliases[identifier]='compare_WMSR_GHRSST_x4_s10'
        else:add(identifier,'loss_sweep','WMSR','GHRSST',l1_weight=weight,ffl_weight=round(1-weight,1))
    path=ROOT/'configs/extended';path.mkdir(parents=True,exist_ok=True)
    for spec in jobs:(path/(spec['id']+'.json')).write_text(json.dumps(spec,indent=2)+'\n')
    manifest={'jobs':jobs,'aliases':aliases,'run_count':len(jobs),'all_original_datasets_unchanged':True,
              'boundary':'Public-block experiments and documented source variants, not recovered private paper code.',
              'loss_sweep_scope':'Fig4 coefficients0/.1/.5/.9/1 verified visually from paper PDF page7',
              'comparison_loss':'Baselines L1; WMSR includes its frequency loss. A method comparison, not an architecture-only causal claim.'}
    (ROOT/'records/extended_plan.json').write_text(json.dumps(manifest,indent=2)+'\n');print('EXTENDED_EXPERIMENTS',len(jobs),'aliases',len(aliases))
if __name__=='__main__':main()
