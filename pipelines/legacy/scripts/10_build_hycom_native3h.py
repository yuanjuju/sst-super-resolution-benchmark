#!/usr/bin/env python3
"""HYCOM GOFS3.1, native3h2016 sources: 1000 HR images, exact800/200."""
import calendar,concurrent.futures,datetime as dt,json
from urllib.parse import urlencode
import xml.etree.ElementTree as ET
import netCDF4,numpy as np
from formal_common import ROOT,fetch,crop_ocean,write_dataset
REGIONS={"NP":(5,45,140,180),"AGM":(-45,-5,-75,-35),
         "NIO":(-35,5,-90,-50),"EWP":(-20,20,-160,-120)}
def main():
    rng=np.random.default_rng(10);raw=ROOT/"data/raw/HYCOM3H";raw.mkdir(exist_ok=True)
    jobs=[]
    for month in range(1,13):
        exp="56.3" if month<=4 else "57.2"
        first=dt.datetime(2016,month,1,12 if month==5 else 0)
        last=dt.datetime(2016,month,calendar.monthrange(2016,month)[1],21)
        for label,(south,north,west,east) in REGIONS.items():
            params={"var":"water_temp","north":north,"south":south,"west":west,"east":east,
                    "horizStride":1,"time_start":first.isoformat()+"Z","time_end":last.isoformat()+"Z",
                    "timeStride":9,"vertCoord":0,"accept":"netcdf4"}
            url="https://ncss.hycom.org/thredds/ncss/grid/GLBv0.08/expt_"+exp+"?"+urlencode(params)
            path=raw/("HYCOM3H_"+exp.replace(".","")+"_"+label+"_"+first.strftime("%Y%m")+".nc")
            jobs.append((label,url,path))
    completed=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(fetch,url,path,150):(label,path) for label,url,path in jobs}
        for future in concurrent.futures.as_completed(futures):
            label,path=futures[future];record=future.result();completed.append((label,path,record))
            print("NATIVE3H_MONTH_REGION",len(completed),"/",len(jobs),label,path.name,flush=True)
    lookup={label:{} for label in REGIONS}
    actual_spacing=[]
    for label,path,record in sorted(completed,key=lambda item:item[1].name):
        with netCDF4.Dataset(path) as ds:
            axis=ds["time"];values=axis[:]
            dates=netCDF4.num2date(values,axis.units,only_use_cftime_datetimes=False)
            actual_spacing.append({"file":record["file"],"count":len(values),"first":str(dates[0]),"last":str(dates[-1])})
            for index,value in enumerate(dates):
                stamp=dt.datetime(value.year,value.month,value.day,value.hour,value.minute,value.second)
                if stamp.year==2016:lookup[label][stamp]=(path,index,record)
    common=sorted(set.intersection(*[set(values) for values in lookup.values()]))
    boundary=dt.datetime(2016,10,19)
    train_pool=[x for x in common if x<boundary];test_pool=[x for x in common if x>=boundary]
    assert len(train_pool)>=200 and len(test_pool)>=50,(len(train_pool),len(test_pool))
    train_stamps=set(train_pool[i] for i in rng.choice(len(train_pool),200,replace=False))
    test_stamps=set(test_pool[i] for i in rng.choice(len(test_pool),50,replace=False))
    parts={"train":[],"test":[]}
    for label in REGIONS:
        for stamp in sorted(train_stamps|test_stamps):
            path,index,record=lookup[label][stamp]
            with netCDF4.Dataset(path) as ds:
                variable=ds["water_temp"];array=np.ma.filled(variable[index,0],np.nan).astype(np.float32)
                lat=np.asarray(ds["lat"][:]);lon=np.asarray(ds["lon"][:])
                crop,y,x=crop_ocean(array,rng)
                units=getattr(variable,"units","")
            assert -3<float(crop.min()) and float(crop.max())<45,(units,float(crop.min()),float(crop.max()))
            split="train" if stamp in train_stamps else "test"
            meta={"sample_id":"HYCOM_"+label+"_"+stamp.strftime("%Y%m%dT%H%M")+"_"+str(y)+"_"+str(x),
                  "region_label":label,"paper_region_bbox":list(REGIONS[label]),
                  "date":stamp.date().isoformat(),"timestamp_utc":stamp.isoformat()+"Z",
                  "source":record["file"],"time_index":index,"depth_m":0,"source_units":units,
                  "row":y,"column":x,"lat_bounds":[float(lat[y]),float(lat[y+95])],
                  "lon_bounds":[float(lon[x]),float(lon[x+95])],
                  "native_lon_spacing_deg":float(np.median(np.diff(lon))),
                  "native_lat_spacing_min_deg":float(np.diff(lat[y:y+96]).min()),
                  "native_lat_spacing_max_deg":float(np.diff(lat[y:y+96]).max())}
            parts[split].append((crop,meta))
    for split in parts:
        order=rng.permutation(len(parts[split]));parts[split]=[parts[split][i] for i in order]
    records=ROOT/"records/formal";records.mkdir(parents=True,exist_ok=True)
    (records/"HYCOM_raw_manifest.json").write_text(json.dumps([x[2] for x in completed],indent=2)+"\n")
    write_dataset("HYCOM",parts,{"product":"HYCOM GOFS3.1 GLBv0.08 expt56.3+57.2",
        "year":2016,"source_depth_m":0,"source_native_time_grid_hours":3,
        "source_time_evidence":"records/hycom_native3h_evidence.json","temporal_interpolation":False,
        "native_source_selection":"Stride9 on the native3h axis obtains candidate snapshots across the entire year; select250 distinct timestamps perregion to obtain the paper's1000-image total",
        "regions":{k:list(v) for k,v in REGIONS.items()},
        "coordinate_choice":"Literal paper numerical coordinates; regional names disagree with geography and are retained as labels only",
        "selected_timestamps_per_region":{"train":200,"test":50},
        "source_candidates_per_region":len(common),"train_test_source_dates_disjoint":True,
        "monthly_source_fields":actual_spacing,"spatial_stride":1,
        "raw_manifest":"records/formal/HYCOM_raw_manifest.json","no_extra_validation_holdout":True})
if __name__=="__main__":main()

