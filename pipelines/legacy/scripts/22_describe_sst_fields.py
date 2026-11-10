#!/usr/bin/env python3
"""Describe retained native SST patches without changing data, splits or models."""
import json
from pathlib import Path
ROOT=Path('/root/wmsr-repro')

def main():
    import numpy as np
    report={}
    for name in ['HYCOM','OISST','GHRSST']:
        manifest=json.loads((ROOT/'records/formal'/f'{name}_manifest.json').read_text())
        entry={}
        for split in ['train','test']:
            with np.load(ROOT/manifest['splits'][split]['file']) as archive:hr=archive['hr_celsius'].astype(np.float64)
            std=hr.std(axis=(1,2))
            gradient=np.sqrt((np.square(np.diff(hr,axis=1)).mean((1,2))+np.square(np.diff(hr,axis=2)).mean((1,2)))/2)
            entry[split]={'count':len(hr),'minimum_c':float(hr.min()),'maximum_c':float(hr.max()),
                          'per_image_std_c_quantiles':dict(zip(['min','p25','median','p75','max'],np.quantile(std,[0,.25,.5,.75,1]).tolist())),
                          'adjacent_pixel_gradient_rms_c_quantiles':dict(zip(['min','p25','median','p75','max'],np.quantile(gradient,[0,.25,.5,.75,1]).tolist()))}
        report[name]=entry
    report['interpretation']='Descriptive statistics on native HR crops. Smoothness alone does not establish why a trained network underperforms bicubic or match the unpublished author image encoding.'
    out=ROOT/'records/formal/field_statistics.json';out.write_text(json.dumps(report,indent=2)+'\n')
    for name in ['HYCOM','OISST','GHRSST']:print(name,json.dumps(report[name]['test']),flush=True)

if __name__=='__main__':main()
