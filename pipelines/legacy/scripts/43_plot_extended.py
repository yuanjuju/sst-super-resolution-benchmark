#!/usr/bin/env python3
"""Fixed-index comparisons across all trained methods; no score-based selection."""
import argparse,json,sys
from pathlib import Path
ROOT=Path('/root/wmsr-repro')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--allow-missing',action='store_true');parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    import numpy as np,torch
    import torch.nn.functional as F
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0,str(ROOT));from extended_models.factory import create_model
    torch.set_num_threads(1);out=ROOT/('records/report_smoke' if args.smoke else 'results/extended_summary');out.mkdir(parents=True,exist_ok=True)
    methods=['EDSR','ELAN','DAT','SAFMN','MambaIR','CRAN','WMSR'];indices=[0,66,133];selection={}
    for dataset in (['GHRSST'] if args.smoke else ['HYCOM','OISST','GHRSST']):
        manifest=json.loads((ROOT/'records/formal'/f'{dataset}_manifest.json').read_text())
        selection[dataset]=[{'test_index':i,'sample_id':manifest['splits']['test']['samples'][i]['sample_id']} for i in indices]
        with np.load(ROOT/manifest['splits']['test']['file']) as a:
            truth=a['hr_celsius'][indices].copy();lows={s:a[f'lr_x{s}_celsius'][indices].copy() for s in [2,3,4]}
        for scale in ([4] if args.smoke else [2,3,4]):
            low=torch.from_numpy((lows[scale]+2)/42*255)[:,None].repeat(1,3,1,1).cuda()
            with torch.inference_mode():bicubic=(F.interpolate(low,scale_factor=scale,mode='bicubic',align_corners=False,antialias=False).mean(1)/255*42-2).cpu().numpy()
            predictions={'Bicubic':bicubic}
            for method in methods:
                run=ROOT/'results/extended'/f'compare_{method}_{dataset}_x{scale}_s10'
                if not (run/'final_results.json').exists():
                    assert args.allow_missing,'Comparison result missing'
                    predictions[method]=None;continue
                checkpoint=torch.load(run/'checkpoints/final.pt',map_location='cpu',weights_only=False)
                model,_=create_model(checkpoint['config']['spec']);model.load_state_dict(checkpoint['model_state_dict']);model=model.cuda().eval()
                with torch.inference_mode():predictions[method]=(model(low).mean(1)/255*42-2).cpu().numpy()
                del model;torch.cuda.empty_cache()
            labels=['HR']+list(predictions);arrays=[truth]+list(predictions.values())
            for error_plot in [False,True]:
                fig,axes=plt.subplots(3,len(labels),figsize=(20,7.1),layout='constrained')
                for row in range(3):
                    low_c=float(truth[row].min());high_c=max(low_c+.001,float(truth[row].max()))
                    max_error=max(.001,max(float(np.abs(value[row]-truth[row]).max()) for value in predictions.values() if value is not None))
                    for col,(label,array) in enumerate(zip(labels,arrays)):
                        ax=axes[row,col]
                        if array is None:
                            ax.text(.5,.5,'FAILED / NO RESULT',ha='center',va='center',transform=ax.transAxes,fontsize=8)
                        else:
                            value=np.abs(array[row]-truth[row]) if error_plot else array[row]
                            plot=ax.imshow(value,cmap='magma' if error_plot else 'viridis',vmin=0 if error_plot else low_c,vmax=max_error if error_plot else high_c)
                        if row==0:ax.set_title(label+('*' if label=='CRAN' else ''),fontsize=10)
                        ax.set_xticks([]);ax.set_yticks([])
                        if col==0:ax.set_ylabel('Test index '+str(indices[row]))
                    fig.colorbar(plot,ax=axes[row,:],shrink=.8,label='Absolute error Celsius' if error_plot else 'SST Celsius')
                fig.suptitle(f'{dataset} x{scale} / fixed indices / reconstructed-data protocol / CRAN*: paper-guided reconstruction',fontsize=12)
                fig.savefig(out/f"{'errors' if error_plot else 'comparison'}_{dataset}_x{scale}.png",dpi=140);plt.close(fig)
            del low;torch.cuda.empty_cache()
    (out/'example_selection.json').write_text(json.dumps({'selection':selection,'rule':'Fixed indices0,66,133, no score-based selection','display':'Whole HR96 patch; mean of3channels in Celsius. Main metrics retain scale-pixel crop.'},indent=2)+'\n')
    print('EXTENDED_COMPARISON_FIGURES_READY',flush=True)
if __name__=='__main__':main()
