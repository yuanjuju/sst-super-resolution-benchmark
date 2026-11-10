"""Shared provenance-preserving helpers for full-size public WMSR datasets."""
from pathlib import Path
import hashlib,json,time,urllib.request
import numpy as np
ROOT=Path("/root/wmsr-repro")

def fetch(url,path,timeout=60):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(4):
        try:
            if not path.exists():
                tmp=path.with_suffix(path.suffix+".part")
                with urllib.request.urlopen(url,timeout=timeout) as response,tmp.open("wb") as out:
                    while chunk:=response.read(1024*1024):out.write(chunk)
                assert tmp.stat().st_size>512,"Empty response"
                tmp.replace(path)
            with path.open("rb") as stream:magic=stream.read(8)
            assert magic.startswith(b"CDF") or magic.startswith(b"\x89HDF"),"Not a NetCDF file"
            return {"file":str(path.relative_to(ROOT)),"url":url,"bytes":path.stat().st_size,
                    "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
        except Exception:
            path.with_suffix(path.suffix+".part").unlink(missing_ok=True)
            if attempt==3:raise
            time.sleep(2**attempt)

def crop_ocean(array,rng,valid=None,size=96):
    valid=np.isfinite(array) if valid is None else (valid & np.isfinite(array))
    invalid=(~valid).astype(np.int32)
    integ=np.pad(invalid,((1,0),(1,0))).cumsum(0).cumsum(1)
    counts=integ[size:,size:]-integ[:-size,size:]-integ[size:,:-size]+integ[:-size,:-size]
    points=np.argwhere(counts==0)
    if not len(points):raise ValueError("No fully valid native-resolution crop")
    y,x=map(int,points[rng.integers(len(points))])
    return array[y:y+size,x:x+size].copy(),y,x

def write_dataset(name,parts,details):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(4)
    assert len(parts["train"])==800 and len(parts["test"])==200
    folder=ROOT/"data/formal"/name;folder.mkdir(parents=True,exist_ok=True)
    records=ROOT/"records/formal";records.mkdir(parents=True,exist_ok=True)
    manifest={"dataset":name,"protocol_version":"paper_size_public_reconstruction_v1",
              "count":1000,"scales":[2,3,4],"seed":10,
              "normalization":{"min_c":-2,"max_c":40,"model_range":255,
                               "clipping":False,"channels":"repeat scalar into 3 channels",
                               "author_rgb_mapping_unavailable":True},
              "hr_size":96,"patch_interpretation":"HR 96 as in released loader; paper input-patch wording ambiguous",
              "downsample":{"mode":"bicubic","antialias":True,"align_corners":False,"torch":torch.__version__},
              "details":details,"splits":{}}
    for split,items in parts.items():
        hr=np.stack([item[0] for item in items]).astype(np.float32)
        assert hr.shape==(len(items),96,96) and np.isfinite(hr).all()
        tensors={"hr_celsius":hr}
        for scale in [2,3,4]:
            tensors["lr_x"+str(scale)+"_celsius"]=F.interpolate(torch.from_numpy(hr)[:,None],scale_factor=1/scale,mode="bicubic",align_corners=False,antialias=True)[:,0].numpy()
        target=folder/(split+".npz")
        with target.with_suffix(".npz.part").open("wb") as stream:np.savez_compressed(stream,**tensors)
        target.with_suffix(".npz.part").replace(target)
        manifest["splits"][split]={"count":len(items),"file":str(target.relative_to(ROOT)),
                                  "sha256":hashlib.sha256(target.read_bytes()).hexdigest(),
                                  "min_c":float(hr.min()),"max_c":float(hr.max()),
                                  "samples":[item[1] for item in items]}
    ids=[item[1]["sample_id"] for values in parts.values() for item in values]
    assert len(set(ids))==1000,"Duplicate image IDs"
    (records/(name+"_manifest.json")).write_text(json.dumps(manifest,indent=2)+"\n")
    print("FULL_DATASET_READY",name,"train=800 test=200 scales=2,3,4",flush=True)

