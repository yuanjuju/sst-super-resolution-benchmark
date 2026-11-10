#!/usr/bin/env python3
"""OISST 2016, global-ocean sampling: exact 800/200 full-size split."""
import datetime as dt,json
import numpy as np
import netCDF4
from formal_common import ROOT,crop_ocean,write_dataset

def main():
    rng=np.random.default_rng(10)
    days=[dt.date(2016,1,1)+dt.timedelta(days=i) for i in range(366)]
    parts={}
    for split,dates,count in [("train",days[:292],800),("test",days[292:],200)]:
        counts=np.full(len(dates),count//len(dates))
        counts[rng.permutation(len(dates))[:count%len(dates)]]+=1
        items=[]
        for day,n in zip(dates,counts):
            source=ROOT/"data/raw"/("oisst-avhrr-v02r01."+day.strftime("%Y%m%d")+".nc")
            with netCDF4.Dataset(source) as ds:
                lat=np.asarray(ds["lat"][:]);lon=np.asarray(ds["lon"][:])
                array=np.ma.filled(ds["sst"][0,0],np.nan).astype(np.float32)
                ice=np.ma.filled(ds["ice"][0,0],np.nan).astype(np.float32)
                units=getattr(ds["ice"],"units","")
                finite_ice=ice[np.isfinite(ice)]
                if finite_ice.size and finite_ice.max()>1.01:ice=ice/100.0
            valid=np.isfinite(array)&~(np.isfinite(ice)&(ice>0.15))
            used=set()
            for j in range(int(n)):
                while True:
                    crop,y,x=crop_ocean(array,rng,valid)
                    if (y,x) not in used:used.add((y,x));break
                meta={"sample_id":"OISST_"+day.strftime("%Y%m%d")+"_"+str(y)+"_"+str(x),
                      "date":day.isoformat(),"source":str(source.relative_to(ROOT)),
                      "row":y,"column":x,"lat_bounds":[float(lat[y]),float(lat[y+95])],
                      "lon_bounds":[float(lon[x]),float(lon[x+95])],"native_resolution_deg":0.25}
                items.append((crop,meta))
            if len(items)%100<4:print("OISST_SAMPLES",split,len(items),flush=True)
        order=rng.permutation(len(items));parts[split]=[items[i] for i in order]
    write_dataset("OISST",parts,{"source":"NOAA OISST v2.1 AVHRR","year":2016,
                  "coverage":"global ocean; 366 original daily NetCDF files retained",
                  "split":"first292 days train / last74 days test; no same-day cross split",
                  "mask":"finite SST; reject observed ice fraction >0.15 where available; missing ice does not invalidate valid SST. Auto-scaled ice already ranges0..1 despite percent unit label. Rejected crops replaced until exact count",
                  "raw_manifest":"records/raw_data_manifest.json",
                  "source_dates_disjoint":True,"no_extra_validation_holdout":True})

if __name__=="__main__":main()

