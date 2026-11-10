#!/usr/bin/env python3
"""Inspect one official OISST file and verify a scalar-temperature x4 data pair."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path("/root/wmsr-repro"))
    args=parser.parse_args()
    root=args.root.resolve()
    import numpy as np
    import netCDF4
    import torch
    import torch.nn.functional as F
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    source=root/"data/raw/oisst-avhrr-v02r01.20160101.nc"
    with netCDF4.Dataset(source) as ds:
        lat=np.asarray(ds.variables["lat"][:])
        lon=np.asarray(ds.variables["lon"][:])
        variable=ds.variables["sst"]
        units=getattr(variable,"units","")
        assert units.lower() in ["celsius","degrees_celsius","degree_celsius","degree_c","degrees_c"],units
        yi=np.flatnonzero((lat>=5)&(lat<=45))
        xi=np.flatnonzero((lon>=140)&(lon<=180))
        values=variable[0,0,int(yi[0]):int(yi[-1])+1,int(xi[0]):int(xi[-1])+1]
        region=np.ma.filled(values,np.nan).astype(np.float32)
        metadata={"dimensions":dict((key,len(value)) for key,value in ds.dimensions.items()),
                  "sst_dimensions":list(variable.dimensions),"sst_units":units,
                  "scale_factor":float(getattr(variable,"scale_factor",1.0)),
                  "add_offset":float(getattr(variable,"add_offset",0.0))}
    patch_size=96
    invalid=(~np.isfinite(region)).astype(np.int32)
    integral=np.pad(invalid,((1,0),(1,0))).cumsum(0).cumsum(1)
    sums=(integral[patch_size:,patch_size:]-integral[:-patch_size,patch_size:]
          -integral[patch_size:,:-patch_size]+integral[:-patch_size,:-patch_size])
    candidates=np.argwhere(sums==0)
    assert len(candidates)>0,"No all-ocean finite 96x96 crop in pilot region"
    center=np.array([(region.shape[0]-patch_size)/2,(region.shape[1]-patch_size)/2])
    y,x=map(int,candidates[np.argmin(((candidates-center)**2).sum(1))])
    hr=region[y:y+patch_size,x:x+patch_size].copy()
    tensor=torch.from_numpy(hr)[None,None]
    lr=F.interpolate(tensor,scale_factor=0.25,mode="bicubic",align_corners=False,antialias=True)
    bicubic=F.interpolate(lr,scale_factor=4,mode="bicubic",align_corners=False,antialias=False)
    difference=(bicubic-tensor)[0,0,4:-4,4:-4].numpy()
    mse=float(np.mean(difference**2))
    report={"stage":"one real-data pilot; no model training",
            "recorded_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "source_file":source.name,"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
            "product":"NOAA OISST v2.1 AVHRR daily","date":"2016-01-01","metadata":metadata,
            "pilot_region":{"latitude":[5,45],"longitude_east":[140,180],
                            "note":"Chosen pilot region; not the unavailable author OISST crop list",
                            "grid_shape":list(region.shape),"finite_fraction":float(np.isfinite(region).mean()),
                            "all_ocean_96px_candidate_count":len(candidates)},
            "crop":{"row":y,"column":x,"shape":list(hr.shape),
                    "sst_min_c":float(hr.min()),"sst_max_c":float(hr.max())},
            "downsample":{"scale":4,"implementation":"torch.nn.functional.interpolate",
                          "mode":"bicubic","align_corners":False,"antialias":True,
                          "torch_version":torch.__version__},
            "lr_shape":list(lr.shape),"upsample_antialias":False,
            "baseline_pilot_only":{"name":"bicubic","crop_border":4,
                                   "rmse_c":float(np.sqrt(mse)),"mae_c":float(np.abs(difference).mean()),
                                   "psnr_db_fixed_42c_range":float(10*np.log10(42**2/mse)),
                                   "limit":"One selected crop, not a test-set score"}}
    (root/"records/first_sst_sample.json").write_text(json.dumps(report,indent=2)+"\n")
    np.savez_compressed(root/"data/first_sample_pair.npz",hr_celsius=hr,lr_celsius=lr[0,0].numpy())
    fig,axes=plt.subplots(1,3,figsize=(11,3.4),layout="constrained")
    vmin,vmax=float(hr.min()),float(hr.max())
    for ax,array,title in zip(axes,[hr,lr[0,0].numpy(),bicubic[0,0].numpy()],
                              ["HR temperature (96 x 96)","Synthetic LR (24 x 24)","Bicubic reconstruction"]):
        im=ax.imshow(array,origin="lower",cmap="viridis",vmin=vmin,vmax=vmax,interpolation="nearest")
        ax.set_title(title,fontsize=11)
        ax.set_xticks([]);ax.set_yticks([])
    fig.colorbar(im,ax=axes,label="SST (degrees Celsius)",shrink=0.84)
    fig.suptitle("OISST pilot crop: data verification before training",fontsize=12)
    fig.savefig(root/"results/first_sst_sample_preview.png",dpi=160)
    plt.close(fig)
    print("FIRST_SST_SAMPLE_PASS",json.dumps(report),flush=True)

if __name__=="__main__":
    main()

