#!/usr/bin/env python3
"""GHRSST/MUR native 0.01-degree SST on 2016-08-12: exact 1000 unique tiles."""
import concurrent.futures,json
from pathlib import Path
from urllib.parse import quote
import numpy as np
import netCDF4
from formal_common import ROOT,fetch,write_dataset
BASE="https://upwell.pfeg.noaa.gov/erddap/griddap/jplMURSST41.nc?"
STAMP="(2016-08-12T09:00:00Z)"
def query(bounds,variables):
    return BASE+quote(",".join(variable+"["+STAMP+"]"+bounds for variable in variables),safe=",():.-_")

def main():
    rng=np.random.default_rng(10)
    raw=ROOT/"data/raw/GHRSST";raw.mkdir(exist_ok=True)
    coarse_url=query("[(-70):100:(70)][(-179.99):100:(180)]",["analysed_sst","mask","sea_ice_fraction"])
    coarse_record=fetch(coarse_url,raw/"coarse_candidate_grid.nc")
    with netCDF4.Dataset(raw/"coarse_candidate_grid.nc") as ds:
        sst=np.ma.filled(ds["analysed_sst"][0],np.nan)
        mask=np.ma.filled(ds["mask"][0],0).astype(int)
        ice=np.ma.filled(ds["sea_ice_fraction"][0],1)
        lat=np.asarray(ds["latitude"][:]);lon=np.asarray(ds["longitude"][:])
    valid=np.isfinite(sst)&((mask&1)>0)&((mask&(2|8|16))==0)
    candidates=np.argwhere(valid)
    candidates=candidates[rng.permutation(len(candidates))]
    accepted=[];rejected=[];raw_records=[coarse_record]
    def get_tile(point):
        a,b=map(int,point)
        row=int(round((float(lat[a])+89.99)/0.01))-48
        col=int(round((float(lon[b])+179.99)/0.01))-48
        if row<0 or row+95>=17999 or col<0 or col+95>=36000:return None
        bounds="["+str(row)+":1:"+str(row+95)+"]["+str(col)+":1:"+str(col+95)+"]"
        url=query(bounds,["analysed_sst","mask","sea_ice_fraction"])
        path=raw/("MUR_20160812_"+str(row)+"_"+str(col)+".nc")
        return fetch(url,path),path,row,col
    position=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        while len(accepted)<1000:
            batch=candidates[position:position+32];position+=len(batch)
            assert len(batch),"Insufficient native ocean tiles"
            futures=[pool.submit(get_tile,point) for point in batch]
            for future in futures:
                result=future.result()
                if result is None:continue
                record,path,row,col=result;raw_records.append(record)
                with netCDF4.Dataset(path) as ds:
                    values=np.ma.filled(ds["analysed_sst"][0],np.nan).astype(np.float32)
                    mask=np.ma.filled(ds["mask"][0],0).astype(int)
                    ice=np.ma.filled(ds["sea_ice_fraction"][0],1)
                    native_lat=np.asarray(ds["latitude"][:]);native_lon=np.asarray(ds["longitude"][:])
                good=(values.shape==(96,96) and np.isfinite(values).all()
                      and np.all((mask&1)>0) and np.all((mask&(2|8|16))==0))
                if good and len(accepted)<1000:
                    accepted.append((values,{"sample_id":"GHRSST_20160812_"+str(row)+"_"+str(col),
                        "date":"2016-08-12","source":record["file"],"row":row,"column":col,
                        "lat_bounds":[float(native_lat[0]),float(native_lat[-1])],
                        "lon_bounds":[float(native_lon[0]),float(native_lon[-1])],"native_resolution_deg":0.01}))
                elif not good:rejected.append(record["file"])
            print("GHRSST_ACCEPTED",len(accepted),"REJECTED",len(rejected),"CANDIDATES",position,flush=True)
    order=rng.permutation(1000)
    parts={"train":[accepted[i] for i in order[:800]],"test":[accepted[i] for i in order[800:]]}
    records=ROOT/"records/formal";records.mkdir(parents=True,exist_ok=True)
    (records/"GHRSST_raw_manifest.json").write_text(json.dumps(raw_records,indent=2)+"\n")
    write_dataset("GHRSST",parts,{"product":"MUR-JPL-L4-GLOB-v4.1","date":"2016-08-12",
        "product_choice":"Paper did not specify GHRSST product; MUR chosen for matching date and native 0.01-degree resolution",
        "coverage":"global ocean between 70S and70N; candidate grid only used to locate tiles",
        "native_sampling":"Each HR tile fetched at stride1, original0.01degree; candidate centers 1degree apart so 96px tiles do not overlap",
        "coarse_grid_is_not_training_data":True,"rejected_tiles_replaced":len(rejected),
        "mask":"all native pixels finite SST and explicit open-sea mask; exclude land/ice flags. Missing sea_ice_fraction at open-water pixels is not rejected",
        "raw_manifest":"records/formal/GHRSST_raw_manifest.json","no_extra_validation_holdout":True})

if __name__=="__main__":main()

