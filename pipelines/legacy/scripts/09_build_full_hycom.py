#!/usr/bin/env python3
"""Full2016 HYCOM source regions and exact250 HR samples per region, 800/200 split."""
import calendar,concurrent.futures,datetime as dt,json
from pathlib import Path
from urllib.parse import urlencode
import xml.etree.ElementTree as ET
import netCDF4
import numpy as np
from formal_common import ROOT,fetch,crop_ocean,write_dataset

# Literal published numeric coordinates; geographical-name inconsistencies are retained.
REGIONS={"NP":(5,45,140,180),"AGM":(-45,-5,-75,-35),
         "NIO":(-35,5,-90,-50),"EWP":(-20,20,-160,-120)}

def main():
    rng=np.random.default_rng(10)
    raw=ROOT/"data/raw/HYCOM";raw.mkdir(exist_ok=True)
    available=set();time_evidence=[]
    for exp,short in [("91.1","911"),("91.2","912")]:
        xml=ET.fromstring((ROOT/"records"/("hycom_"+short+".xml")).read_text())
        axis=xml.find("axis[@name='time']")
        attributes={x.attrib["name"]:x.attrib.get("value") for x in axis.findall("attribute")}
        values=np.fromstring(axis.find("values").text,sep=" ")
        dates=netCDF4.num2date(values,attributes["units"],only_use_cftime_datetimes=False)
        for value in dates:
            day=dt.date(value.year,value.month,value.day)
            if day.year==2016 and ((exp=="91.1" and day<dt.date(2016,4,18)) or (exp=="91.2" and day>=dt.date(2016,4,18))):
                available.add(day)
        time_evidence.append({"experiment":exp,"axis_count":len(values),"units":attributes["units"],
                              "median_spacing":float(np.median(np.diff(values)))})
    dates=sorted(available)
    cut=int(len(dates)*0.8)
    train_dates=sorted(rng.choice(dates[:cut],200,replace=False))
    test_dates=sorted(rng.choice(dates[cut:],50,replace=False))
    train_set=set(train_dates);test_set=set(test_dates);chosen=train_set|test_set
    jobs=[]
    for month in range(1,13):
        first=dt.date(2016,month,1);last=dt.date(2016,month,calendar.monthrange(2016,month)[1])
        spans=[("91.1",first,last)] if month<4 else [("91.2",first,last)]
        if month==4:spans=[("91.1",first,dt.date(2016,4,17)),("91.2",dt.date(2016,4,18),last)]
        for exp,start,end in spans:
            for name,(south,north,west,east) in REGIONS.items():
                params={"var":"water_temp","north":north,"south":south,"west":west,"east":east,
                        "horizStride":1,"time_start":start.isoformat()+"T00:00:00Z",
                        "time_end":end.isoformat()+"T00:00:00Z","timeStride":1,"vertCoord":0,"accept":"netcdf4"}
                url="https://ncss.hycom.org/thredds/ncss/grid/GLBu0.08/expt_"+exp+"/ts3z?"+urlencode(params)
                path=raw/("HYCOM_"+exp.replace(".","")+"_"+name+"_"+start.strftime("%Y%m%d")+"_"+end.strftime("%Y%m%d")+".nc")
                jobs.append((name,url,path))
    completed=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(fetch,url,path,150):(name,path) for name,url,path in jobs}
        for future in concurrent.futures.as_completed(futures):
            record=future.result();name,path=futures[future]
            completed.append((name,path,record))
            print("HYCOM_MONTH_REGION",len(completed),"/",len(jobs),name,path.name,flush=True)
    parts={"train":[],"test":[]};counts={}
    for name,path,record in sorted(completed,key=lambda x:x[1].name):
        with netCDF4.Dataset(path) as ds:
            time=ds["time"];timestamps=netCDF4.num2date(time[:],time.units,only_use_cftime_datetimes=False)
            lat=np.asarray(ds["lat"][:]);lon=np.asarray(ds["lon"][:])
            variable=ds["water_temp"]
            for index,value in enumerate(timestamps):
                day=dt.date(value.year,value.month,value.day)
                if day not in chosen:continue
                key=(name,day.isoformat())
                if key in counts:continue
                array=np.ma.filled(variable[index,0],np.nan).astype(np.float32)
                hr,y,x=crop_ocean(array,rng)
                split="train" if day in train_set else "test"
                metadata={"sample_id":"HYCOM_"+name+"_"+day.strftime("%Y%m%d")+"_"+str(y)+"_"+str(x),
                          "region_label":name,"paper_region_bbox":list(REGIONS[name]),
                          "date":day.isoformat(),"source":record["file"],"time_index":index,
                          "row":y,"column":x,"depth_m":0,"native_resolution_deg":0.08,
                          "lat_bounds":[float(lat[y]),float(lat[y+95])],
                          "lon_bounds":[float(lon[x]),float(lon[x+95])]}
                parts[split].append((hr,metadata));counts[key]=True
    assert len(counts)==1000,(len(counts),len(parts["train"]),len(parts["test"]))
    for split in parts:
        order=rng.permutation(len(parts[split]));parts[split]=[parts[split][i] for i in order]
    records=ROOT/"records/formal";records.mkdir(parents=True,exist_ok=True)
    (records/"HYCOM_raw_manifest.json").write_text(json.dumps([x[2] for x in completed],indent=2)+"\n")
    write_dataset("HYCOM",parts,{"product":"HYCOM GOFS3.0 GLBu0.08 expt91.1+91.2",
        "year":2016,"source_depth_m":0,"regions":{k:list(v) for k,v in REGIONS.items()},
        "coordinate_choice":"Use literal paper numerical coordinates; AGM/NIO/EWP names conflict with geography",
        "source_available_days":len(dates),"source_time_axis_evidence":time_evidence,
        "temporal_limitation":"Public2016 hindcast aggregates are daily24h, not the paper's3h. No temporal interpolation or duplicated frames. Sample count remains1000.",
        "sampling":"250 distinct source dates per region;200train+50test per region; source dates disjoint acrosssplit",
        "full_native_source":"All available2016 monthly SST surface fields for each published region are retained at stride1",
        "raw_manifest":"records/formal/HYCOM_raw_manifest.json","no_extra_validation_holdout":True})

if __name__=="__main__":main()

