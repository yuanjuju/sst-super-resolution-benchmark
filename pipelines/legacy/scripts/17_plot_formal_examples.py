#!/usr/bin/env python3
"""Fixed-index SST comparisons; no selection using test scores."""
import hashlib,json,os,sys
from pathlib import Path

ROOT=Path('/root/wmsr-repro')

def main():
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import numpy as np
    import torch
    import torch.nn.functional as F
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0,str(ROOT/'upstream'));sys.path.insert(0,str(ROOT))
    from paper_models.wmsr_paper import WMSRPaper
    torch.set_num_threads(1)
    out=ROOT/'results/summary';out.mkdir(exist_ok=True)
    chosen=[0,66,133]
    selection={'rule':'Fixed test indices [0,66,133] for every dataset and scale; no metric-based selection',
               'display':'Mean of three predicted channels, inverse fixed -2..40C mapping; full uncropped patch. Error colors share scale within a row.',
               'samples':{}}
    for name in ['HYCOM','OISST','GHRSST']:
        manifest=json.loads((ROOT/'records/formal'/f'{name}_manifest.json').read_text())
        part=manifest['splits']['test']
        data_path=ROOT/part['file']
        assert hashlib.sha256(data_path.read_bytes()).hexdigest()==part['sha256']
        with np.load(data_path) as archive:
            truth=archive['hr_celsius'][chosen].copy()
            low={s:archive[f'lr_x{s}_celsius'][chosen].copy() for s in [2,3,4]}
        selection['samples'][name]=[{'test_index':i,'sample_id':part['samples'][i]['sample_id']} for i in chosen]
        for scale in [2,3,4]:
            run=ROOT/'results/formal'/f'{name}_x{scale}_seed10'
            checkpoint=torch.load(run/'checkpoints/final.pt',map_location='cpu')
            assert checkpoint['epoch']==100 and checkpoint['step']==10000
            model=WMSRPaper(scale=scale).cuda().eval()
            model.load_state_dict(checkpoint['model_state_dict'])
            inputs=torch.from_numpy((low[scale]+2)/42*255)[:,None].repeat(1,3,1,1).cuda()
            with torch.inference_mode():
                sr=(model(inputs).mean(1)/255*42-2).cpu().numpy()
                bic=(F.interpolate(inputs,scale_factor=scale,mode='bicubic',align_corners=False,
                                   antialias=False).mean(1)/255*42-2).cpu().numpy()
            assert np.isfinite(sr).all()
            fig,axes=plt.subplots(3,5,figsize=(13.2,8.3),layout='constrained')
            for row,(hr,baseline,prediction) in enumerate(zip(truth,bic,sr)):
                error_b=np.abs(baseline-hr);error_m=np.abs(prediction-hr)
                values=[hr,baseline,prediction,error_b,error_m]
                low_c=float(hr.min());high_c=max(low_c+.01,float(hr.max()))
                high_error=max(.001,float(max(error_b.max(),error_m.max())))
                for col,value in enumerate(values):
                    ax=axes[row,col]
                    plot=ax.imshow(value,cmap='viridis' if col<3 else 'magma',
                                   vmin=low_c if col<3 else 0,vmax=high_c if col<3 else high_error)
                    ax.set_xticks([]);ax.set_yticks([])
                    if row==0:ax.set_title(['HR SST','Bicubic SST','Reconstructed SST',
                                            'Bicubic absolute error','Model absolute error'][col],fontsize=10)
                    if col==0:ax.set_ylabel('Test index '+str(chosen[row]))
                    if col in [2,4]:fig.colorbar(plot,ax=axes[row,:3] if col==2 else axes[row,3:],shrink=.79,label='Celsius')
            fig.suptitle(f'{name} x{scale} / fixed examples / approximate public reconstruction',fontsize=13)
            fig.savefig(out/f'examples_{name}_x{scale}.png',dpi=150)
            plt.close(fig)
            del model,inputs
            torch.cuda.empty_cache()
    (out/'example_selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    print('FIXED_EXAMPLE_FIGURES_READY',flush=True)

if __name__=='__main__':main()
