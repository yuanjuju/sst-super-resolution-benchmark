#!/usr/bin/env python3
"""Train the pinned public WMSR on a recorded, reconstructed scalar-SST protocol."""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace

def save_json(path,value):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2)+"\n")
    temporary.replace(path)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path("/root/wmsr-repro"))
    parser.add_argument("--epochs",type=int,default=100)
    parser.add_argument("--resume",type=Path)
    args=parser.parse_args()
    root=args.root.resolve()
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader,Dataset
    import yaml
    from focal_frequency_loss import FocalFrequencyLoss
    from pytorch_msssim import ssim
    sys.path.insert(0,str(root/"upstream"))
    from models.wmsr_network import create_model

    seed=10
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True
    assert torch.cuda.is_available()
    manifest_path=root/"records/dataset_manifest.json"
    manifest=json.loads(manifest_path.read_text())
    config=yaml.safe_load((root/"upstream/configs/wmsr_x4.yml").read_text())
    config.update({"epochs":args.epochs,"data_repeat":1,"seed":seed,"eval_every":5,
                   "normalization_c":[-2.0,40.0],"evaluation_channel":"mean of 3 predicted channels",
                   "prediction_clipping":False,"dataset_protocol":manifest["protocol"],
                   "dataset_manifest_sha256":hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                   "checkpoint_selection":"lowest validation RMSE in Celsius",
                   "upstream_commit":subprocess.check_output(["git","-C",str(root/"upstream"),"rev-parse","HEAD"],text=True).strip(),
                   "differences":"Reconstructed scalar OISST crops/split; public 4-WAM model; 100 epochs, repeat 1; no claim of paper-table alignment"})
    run=root/"results/oisst_x4_seed10";run.mkdir(exist_ok=True)
    checkpoints=run/"checkpoints";checkpoints.mkdir(exist_ok=True)
    save_json(run/"config.json",config)
    save_json(root/"configs/oisst_x4_public_v1.json",config)

    class SST(Dataset):
        def __init__(self,split,augment=False):
            part=manifest["splits"][split]
            source=root/part["file"]
            assert hashlib.sha256(source.read_bytes()).hexdigest()==part["sha256"]
            with np.load(source) as arrays:
                self.hr=arrays["hr_celsius"].copy();self.lr=arrays["lr_celsius"].copy()
            self.augment=augment
            self.samples=part["samples"]
        def __len__(self):return len(self.hr)
        def __getitem__(self,i):
            lr=self.lr[i];hr=self.hr[i]
            if self.augment:
                if random.random()<0.5: lr=lr[:,::-1];hr=hr[:,::-1]
                if random.random()<0.5: lr=lr[::-1,:];hr=hr[::-1,:]
                if random.random()<0.5: lr=lr.T;hr=hr.T
            lr=torch.from_numpy(((lr+2.0)/42.0*255.0).copy())[None].repeat(3,1,1)
            hr=torch.from_numpy(((hr+2.0)/42.0*255.0).copy())[None].repeat(3,1,1)
            return lr,hr

    generator=torch.Generator().manual_seed(seed)
    train_dataset=SST("train",True);val_dataset=SST("val")
    train=DataLoader(train_dataset,batch_size=8,shuffle=True,drop_last=True,num_workers=0,
                     pin_memory=True,generator=generator)
    val=DataLoader(val_dataset,batch_size=8,shuffle=False,num_workers=0,pin_memory=True)
    model=create_model(SimpleNamespace(**config)).cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=config["lr"])
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=config["decays"],gamma=config["gamma"])
    frequency_loss=FocalFrequencyLoss(loss_weight=1.0,alpha=1.0)

    @torch.inference_mode()
    def evaluate(loader,predictor):
        rows=[]
        for lr,hr in loader:
            lr=lr.cuda(non_blocking=True);hr=hr.cuda(non_blocking=True)
            sr=predictor(lr).mean(dim=1,keepdim=True)
            truth=hr.mean(dim=1,keepdim=True)
            sr=sr[:,:,4:-4,4:-4];truth=truth[:,:,4:-4,4:-4]
            errors=(sr-truth)/255.0*42.0
            mse=errors.square().flatten(1).mean(1)
            mae=errors.abs().flatten(1).mean(1)
            psnr=10*torch.log10(42.0**2/mse.clamp_min(1e-16))
            structural=ssim(sr,truth,data_range=255.0,size_average=False)
            for values in zip(mse.tolist(),mae.tolist(),psnr.tolist(),structural.tolist()):
                rows.append({"mse_c2":values[0],"mae_c":values[1],"psnr_db":values[2],"ssim":values[3]})
        total={"count":len(rows),"rmse_c":math.sqrt(sum(x["mse_c2"] for x in rows)/len(rows)),
               "mae_c":sum(x["mae_c"] for x in rows)/len(rows),
               "psnr_db":sum(x["psnr_db"] for x in rows)/len(rows),
               "ssim":sum(x["ssim"] for x in rows)/len(rows)}
        assert all(math.isfinite(x) for x in total.values())
        return total,rows

    bicubic=lambda x:F.interpolate(x,scale_factor=4,mode="bicubic",align_corners=False,antialias=False)
    baseline_val,_=evaluate(val,bicubic)
    save_json(run/"bicubic_validation.json",baseline_val)
    print("BICUBIC_VALIDATION",json.dumps(baseline_val),flush=True)
    start_epoch=1;step=0;best_rmse=float("inf");best_epoch=0
    if args.resume:
        checkpoint=torch.load(args.resume,map_location="cuda")
        assert checkpoint["config"]["dataset_manifest_sha256"]==config["dataset_manifest_sha256"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch=checkpoint["epoch"]+1;step=checkpoint["step"]
        best_rmse=checkpoint["best_rmse_c"];best_epoch=checkpoint["best_epoch"]
        random.setstate(checkpoint["rng_python"]);np.random.set_state(checkpoint["rng_numpy"])
        torch.set_rng_state(checkpoint["rng_torch"].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in checkpoint["rng_cuda"]])
        generator.set_state(checkpoint["rng_loader"].cpu())
        print("RESUMED",start_epoch,step,flush=True)
    fields=["epoch","step","lr","train_loss","train_l1","train_ffl","val_rmse_c","val_mae_c","val_psnr_db","val_ssim","epoch_seconds","elapsed_seconds"]
    csv_path=run/"training_metrics.csv"
    append=bool(args.resume and csv_path.exists())
    started=time.monotonic()
    with csv_path.open("a" if append else "w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        if not append:writer.writeheader()
        for epoch in range(start_epoch,args.epochs+1):
            model.train();epoch_start=time.monotonic();sums=np.zeros(3);lr_used=optimizer.param_groups[0]["lr"]
            for batch_index,(lr,hr) in enumerate(train):
                lr=lr.cuda(non_blocking=True);hr=hr.cuda(non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                sr=model(lr)
                l1=F.l1_loss(sr,hr);ffl=frequency_loss(sr,hr)
                loss=0.9*l1+0.1*ffl
                if not torch.isfinite(loss):raise RuntimeError("Nonfinite loss at step "+str(step))
                loss.backward()
                if step%100==0:
                    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                optimizer.step();step+=1
                sums+=np.array([loss.item(),l1.item(),ffl.item()])
                if step%100==0:print("STEP",step,"EPOCH",epoch,"LOSS",round(loss.item(),5),flush=True)
            scheduler.step()
            metrics={}
            if epoch==1 or epoch%5==0 or epoch==args.epochs:
                model.eval();metrics,_=evaluate(val,model)
                if metrics["rmse_c"]<best_rmse:
                    best_rmse=metrics["rmse_c"];best_epoch=epoch
                    compact={"model_state_dict":model.state_dict(),"epoch":epoch,"step":step,
                             "config":config,"validation":metrics}
                    temporary=checkpoints/"best.pt.tmp";torch.save(compact,temporary);temporary.replace(checkpoints/"best.pt")
                print("VALIDATION",epoch,json.dumps(metrics),"BEST_EPOCH",best_epoch,flush=True)
            row={"epoch":epoch,"step":step,"lr":lr_used,
                 "train_loss":sums[0]/len(train),"train_l1":sums[1]/len(train),"train_ffl":sums[2]/len(train),
                 "epoch_seconds":time.monotonic()-epoch_start,"elapsed_seconds":time.monotonic()-started}
            row.update({"val_"+k:v for k,v in metrics.items() if k!="count"})
            writer.writerow(row);stream.flush()
            last={"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),
                  "scheduler_state_dict":scheduler.state_dict(),"epoch":epoch,"step":step,
                  "best_rmse_c":best_rmse,"best_epoch":best_epoch,"config":config,
                  "rng_python":random.getstate(),"rng_numpy":np.random.get_state(),
                  "rng_torch":torch.get_rng_state(),"rng_cuda":torch.cuda.get_rng_state_all(),
                  "rng_loader":generator.get_state()}
            temporary=checkpoints/"last.pt.tmp";torch.save(last,temporary);temporary.replace(checkpoints/"last.pt")
            save_json(run/"progress.json",{"status":"training","epoch":epoch,"epochs":args.epochs,
                       "step":step,"best_epoch":best_epoch,"best_validation_rmse_c":best_rmse,
                       "last_epoch":row,"baseline_validation":baseline_val})

    best=torch.load(checkpoints/"best.pt",map_location="cuda")
    assert set(best["model_state_dict"])==set(model.state_dict())
    model.load_state_dict(best["model_state_dict"]);model.eval()
    test_dataset=SST("test")
    test=DataLoader(test_dataset,batch_size=8,shuffle=False,num_workers=0,pin_memory=True)
    wmsr_test,wmsr_rows=evaluate(test,model)
    bicubic_test,bicubic_rows=evaluate(test,bicubic)
    for name,rows in [("wmsr",wmsr_rows),("bicubic",bicubic_rows)]:
        with (run/(name+"_test_per_sample.csv")).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=["sample_id","date","mse_c2","mae_c","psnr_db","ssim"])
            writer.writeheader()
            for sample,values in zip(test_dataset.samples,rows):
                writer.writerow({"sample_id":sample["sample_id"],"date":sample["date"],**values})
    result={"status":"complete","completed_at":dt.datetime.now(dt.timezone.utc).isoformat(),
            "protocol":"oisst_np_x4_public_v1","config":config,"epochs_completed":args.epochs,
            "optimizer_steps":step,"best_epoch":best["epoch"],"best_validation":best["validation"],
            "test_wmsr":wmsr_test,"test_bicubic":bicubic_test,
            "delta_psnr_db":wmsr_test["psnr_db"]-bicubic_test["psnr_db"],
            "rmse_reduction_percent":100*(1-wmsr_test["rmse_c"]/bicubic_test["rmse_c"]),
            "training_and_evaluation_seconds":time.monotonic()-started,
            "limits":"One seed, one reconstructed region/split, synthetic bicubic x4; public implementation differs from paper architecture and original data. Test set evaluated after selecting model on validation RMSE."}
    save_json(run/"final_results.json",result)
    save_json(run/"progress.json",result)
    print("TRAINING_AND_TEST_COMPLETE",json.dumps(result),flush=True)

if __name__=="__main__":
    main()

