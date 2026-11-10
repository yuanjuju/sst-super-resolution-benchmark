#!/usr/bin/env python3
"""Read-only acceptance checks for full dataset counts, uniqueness and raw provenance."""
import hashlib,json
from pathlib import Path
import numpy as np,netCDF4
ROOT=Path("/root/wmsr-repro")
def main():
    checked={}
    for name in ["OISST","GHRSST","HYCOM"]:
        manifest=json.loads((ROOT/"records/formal"/(name+"_manifest.json")).read_text())
        content=[];ids=[];dates={};quality={};source_checks=[]
        for split,expected in [("train",800),("test",200)]:
            part=manifest["splits"][split];path=ROOT/part["file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest()==part["sha256"]
            with np.load(path) as arrays:
                hr=arrays["hr_celsius"]
                assert hr.shape==(expected,96,96) and np.isfinite(hr).all()
                content.extend(hashlib.sha256(x.tobytes()).hexdigest() for x in hr)
                ids.extend(x["sample_id"] for x in part["samples"])
                dates[split]=set(x["date"] for x in part["samples"])
                shapes={}
                for scale in [2,3,4]:
                    lr=arrays["lr_x"+str(scale)+"_celsius"]
                    assert lr.shape==(expected,96//scale,96//scale) and np.isfinite(lr).all()
                    shapes[str(scale)]=list(lr.shape)
                for index in [0,expected//2,expected-1]:
                    meta=part["samples"][index]
                    with netCDF4.Dataset(ROOT/meta["source"]) as ds:
                        if name=="OISST":
                            raw=ds["sst"][0,0,meta["row"]:meta["row"]+96,meta["column"]:meta["column"]+96]
                        elif name=="GHRSST":raw=ds["analysed_sst"][0]
                        else:raw=ds["water_temp"][meta["time_index"],0,meta["row"]:meta["row"]+96,meta["column"]:meta["column"]+96]
                        raw=np.ma.filled(raw,np.nan).astype(np.float32)
                    assert np.array_equal(raw,hr[index]),("Raw source/crop mismatch",name,split,index)
                    source_checks.append({"sample_id":meta["sample_id"],"raw_crop_exact_match":True})
                quality[split]={"count":expected,"hr_shape":list(hr.shape),"lr_shapes":shapes,"all_finite":True}
        assert len(ids)==1000 and len(set(ids))==1000 and len(set(content))==1000
        if name!="GHRSST":assert not (dates["train"]&dates["test"])
        else:
            train=manifest["splits"]["train"]["samples"];test=manifest["splits"]["test"]["samples"]
            positions=np.array([[x["row"],x["column"]] for x in train])
            for meta in test:
                assert not np.any(np.all(np.abs(positions-[meta["row"],meta["column"]])<96,axis=1))
        checked[name]={"splits":quality,"unique_hr_content":1000,"unique_image_ids":1000,
                       "hr_tensor_sha256_by_sample":dict(zip(ids,content)),
                       "raw_provenance_samples_checked":source_checks,
                       "split_separation":"spatial tiles nonoverlap" if name=="GHRSST" else "source dates disjoint"}
    path=ROOT/"records/formal/data_quality_checked.json"
    path.write_text(json.dumps(checked,indent=2)+"\n")
    print("ALL_FULL_DATASETS_VERIFIED",json.dumps({k:{"unique_hr":1000,"train":800,"test":200} for k in checked}),flush=True)
if __name__=="__main__":main()

