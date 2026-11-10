#!/usr/bin/env python3
"""Build an explicitly reconstructed OISST x4 protocol from NOAA daily NetCDFs."""
import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
from pathlib import Path
import time
import urllib.request

BASE="https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr"
def download(day,raw):
    name="oisst-avhrr-v02r01."+day.strftime("%Y%m%d")+".nc"
    url=BASE+"/"+day.strftime("%Y%m")+"/"+name
    path=raw/name
    for attempt in range(4):
        try:
            if not path.exists():
                part=path.with_suffix(".nc.part")
                with urllib.request.urlopen(url,timeout=35) as response,part.open("wb") as target:
                    while chunk:=response.read(1024*1024): target.write(chunk)
                assert part.stat().st_size>10000,"Truncated NetCDF"
                part.replace(path)
            signature=path.read_bytes()[:8]
            assert signature.startswith(b"\x89HDF") or signature.startswith(b"CDF"),"Unexpected file signature"
            return {"date":day.isoformat(),"file":name,"url":url,
                    "bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
        except Exception:
            path.with_suffix(".nc.part").unlink(missing_ok=True)
            if attempt==3: raise
            time.sleep(2**attempt)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path("/root/wmsr-repro"))
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    root=args.root.resolve()
    raw=root/"data/raw"; raw.mkdir(exist_ok=True)
    output=root/"data/oisst_np_x4"; output.mkdir(exist_ok=True)
    started=time.monotonic()
    days=[dt.date(2016,1,1)+dt.timedelta(days=i) for i in range(366)]
    raw_records=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(download,day,raw) for day in days]
        for future in concurrent.futures.as_completed(futures):
            raw_records.append(future.result())
            if len(raw_records)%25==0 or len(raw_records)==len(days):
                print("DOWNLOAD",len(raw_records),"/",len(days),flush=True)
    raw_records.sort(key=lambda value:value["date"])
    (root/"records/raw_data_manifest.json").write_text(json.dumps(raw_records,indent=2)+"\n")

    import numpy as np
    import netCDF4
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(4)
    regions={}
    latitude=None; longitude=None
    for item in raw_records:
        with netCDF4.Dataset(raw/item["file"]) as ds:
            lat=np.asarray(ds.variables["lat"][:]); lon=np.asarray(ds.variables["lon"][:])
            yi=np.flatnonzero((lat>=5)&(lat<=45)); xi=np.flatnonzero((lon>=140)&(lon<=180))
            variable=ds.variables["sst"]
            units=getattr(variable,"units","").lower()
            assert units in ["celsius","degrees_celsius","degree_celsius","degree_c","degrees_c"],units
            array=variable[0,0,int(yi[0]):int(yi[-1])+1,int(xi[0]):int(xi[-1])+1]
            regions[item["date"]]=np.ma.filled(array,np.nan).astype(np.float32)
            latitude=lat[yi];longitude=lon[xi]
    rng=np.random.default_rng(10)
    splits={"train":(days[:262],720),"val":(days[262:292],80),"test":(days[292:],200)}
    manifest={"protocol":"oisst_np_x4_public_v1","year":2016,"product":"NOAA OISST v2.1 AVHRR daily",
              "region":{"latitude":[5,45],"longitude_east":[140,180],
                        "reason":"Explicit pilot region; author OISST crop list unavailable"},
              "scale":4,"hr_patch_size":96,"lr_patch_size":24,"seed":10,
              "mask_policy":"Only fully finite SST crops; no additional ice/quality filter",
              "temperature_storage":"float32 Celsius; netCDF4 automatic mask and scale",
              "normalization":{"min_c":-2.0,"max_c":40.0,"model_range":255.0,
                               "formula":"(sst_celsius + 2) / 42 * 255","clipping":False,
                               "input_channels":"same scalar repeated into three channels"},
              "downsample":{"implementation":"torch.nn.functional.interpolate","torch":torch.__version__,
                            "mode":"bicubic","align_corners":False,"antialias":True},
              "evaluation":{"crop_border":4,"channel":"mean of 3 predicted channels",
                            "prediction_clipping":False,"psnr_fixed_temperature_range_c":42.0},
              "raw_manifest":"records/raw_data_manifest.json","splits":{}}
    for split,(dates,count) in splits.items():
        per_date=np.full(len(dates),count//len(dates),dtype=int)
        per_date[rng.permutation(len(dates))[:count%len(dates)]]+=1
        images=[];samples=[]
        for day,n in zip(dates,per_date):
            region=regions[day.isoformat()]
            invalid=(~np.isfinite(region)).astype(np.int32)
            integral=np.pad(invalid,((1,0),(1,0))).cumsum(0).cumsum(1)
            size=96
            bad=(integral[size:,size:]-integral[:-size,size:]
                 -integral[size:,:-size]+integral[:-size,:-size])
            candidates=np.argwhere(bad==0)
            assert len(candidates)>=n,(day,len(candidates),n)
            chosen=candidates[rng.choice(len(candidates),size=n,replace=False)]
            for y,x in chosen:
                y=int(y);x=int(x)
                images.append(region[y:y+size,x:x+size].copy())
                samples.append({"sample_id":split+"_"+day.strftime("%Y%m%d")+"_"+str(y)+"_"+str(x),
                                "date":day.isoformat(),"row":y,"column":x,
                                "lat_bounds":[float(latitude[y]),float(latitude[y+size-1])],
                                "lon_bounds":[float(longitude[x]),float(longitude[x+size-1])]})
        order=rng.permutation(len(images))
        hr=np.stack(images)[order]
        samples=[samples[i] for i in order]
        assert hr.shape==(count,96,96) and np.isfinite(hr).all()
        lr=F.interpolate(torch.from_numpy(hr)[:,None],scale_factor=0.25,mode="bicubic",
                         align_corners=False,antialias=True)[:,0].numpy()
        target=output/(split+".npz")
        with target.with_suffix(".npz.part").open("wb") as f:
            np.savez_compressed(f,hr_celsius=hr,lr_celsius=lr)
        target.with_suffix(".npz.part").replace(target)
        manifest["splits"][split]={"count":count,"date_start":dates[0].isoformat(),
                                  "date_end":dates[-1].isoformat(),"source_day_count":len(dates),
                                  "file":str(target.relative_to(root)),
                                  "sha256":hashlib.sha256(target.read_bytes()).hexdigest(),
                                  "hr_min_c":float(hr.min()),"hr_max_c":float(hr.max()),
                                  "samples":samples}
        print("DATASET",split,count,"HR",list(hr.shape),"LR",list(lr.shape),flush=True)
    date_sets=[set(sample["date"] for sample in part["samples"]) for part in manifest["splits"].values()]
    assert not (date_sets[0]&date_sets[1] or date_sets[0]&date_sets[2] or date_sets[1]&date_sets[2])
    manifest["source_dates_disjoint"]=True
    manifest["elapsed_seconds"]=time.monotonic()-started
    (root/"records/dataset_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("DATASET_BUILD_COMPLETE",round(manifest["elapsed_seconds"],2),"seconds",flush=True)

if __name__=="__main__":
    main()

