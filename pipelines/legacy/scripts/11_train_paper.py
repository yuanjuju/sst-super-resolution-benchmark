#!/usr/bin/env python3
"""100-epoch paper-size experiment with explicit topology/encoding assumptions."""
import argparse,csv,datetime as dt,hashlib,json,math,os,random,subprocess,sys,time
from pathlib import Path
ROOT=Path("/root/wmsr-repro")
def json_save(path,value):
    temp=path.with_suffix(path.suffix+".tmp");temp.write_text(json.dumps(value,indent=2)+"\n");temp.replace(path)
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",choices=["OISST","GHRSST","HYCOM"],required=True)
    parser.add_argument("--scale",type=int,choices=[2,3,4],required=True)
    parser.add_argument("--epochs",type=int,default=100)
    args=parser.parse_args()
    import numpy as np,torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset,DataLoader
    from focal_frequency_loss import FocalFrequencyLoss
    from pytorch_msssim import ssim
    sys.path.insert(0,str(ROOT/"upstream"));sys.path.insert(0,str(ROOT))
    from paper_models.wmsr_paper import WMSRPaper,ARCHITECTURE_NOTES
    from utils import rgb_to_ycbcr
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=True
    random.seed(10);np.random.seed(10);torch.manual_seed(10);torch.cuda.manual_seed_all(10)
    manifest_path=ROOT/"records/formal"/(args.dataset+"_manifest.json")
    manifest=json.loads(manifest_path.read_text())
    assert manifest["count"]==1000 and manifest["splits"]["train"]["count"]==800 and manifest["splits"]["test"]["count"]==200
    run=ROOT/"results/formal"/(args.dataset+"_x"+str(args.scale)+"_seed10")
    run.mkdir(parents=True,exist_ok=True);(run/"checkpoints").mkdir(exist_ok=True)
    config={"dataset":args.dataset,"scale":args.scale,"epochs":args.epochs,"batch_size":8,"seed":10,
            "train_count":800,"test_count":200,"hr_patch":96,"lr_patch":96//args.scale,
            "repeat":1,"optimizer":"Adam","lr":1e-3,"betas":[0.9,0.999],"eps":1e-8,
            "scheduler":{"type":"StepLR","step_size":20,"gamma":0.5,
                         "paper_ambiguity":"Paper also says cosine annealing; use explicit every20epochs half"},
            "loss":{"L1":0.1,"FFL":0.9,"FFL_alpha":1},
            "architecture":ARCHITECTURE_NOTES,
            "model_source_sha256":hashlib.sha256((ROOT/"paper_models/wmsr_paper.py").read_bytes()).hexdigest(),
            "dataset_manifest_sha256":hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "evaluation":{"image":"clamp0..255 then BT601 Y; per-image PSNR/SSIM averaged; no biased rounding",
                          "physical":"unclipped mean of3channels inverse fixed -2..40C normalization",
                          "crop_border":args.scale,"checkpoint":"fixed final epoch; no test-based model selection"},
            "protocol_limit":"Rebuilt public datasets and inferred missing architecture details; not author-confirmed exact reproduction"}
    if (run/"final_results.json").exists():
        print("ALREADY_COMPLETE",run,flush=True);return
    if (run/"config.json").exists():
        old=json.loads((run/"config.json").read_text())
        assert old==config,"Cannot silently change existing experiment protocol"
    json_save(run/"config.json",config)
    (run/"model_source.py").write_bytes((ROOT/"paper_models/wmsr_paper.py").read_bytes())

    class SST(Dataset):
        def __init__(self,split,augment=False):
            part=manifest["splits"][split];source=ROOT/part["file"]
            assert hashlib.sha256(source.read_bytes()).hexdigest()==part["sha256"]
            with np.load(source) as a:
                self.hr=a["hr_celsius"].copy();self.lr=a["lr_x"+str(args.scale)+"_celsius"].copy()
            self.samples=part["samples"];self.augment=augment
        def __len__(self):return len(self.hr)
        def __getitem__(self,i):
            lr=self.lr[i];hr=self.hr[i]
            if self.augment:
                if random.random()<.5:lr=lr[:,::-1];hr=hr[:,::-1]
                if random.random()<.5:lr=lr[::-1,:];hr=hr[::-1,:]
                if random.random()<.5:lr=lr.T;hr=hr.T
            return (torch.from_numpy(((lr+2)/42*255).copy())[None].repeat(3,1,1),
                    torch.from_numpy(((hr+2)/42*255).copy())[None].repeat(3,1,1))
    generator=torch.Generator().manual_seed(10)
    train_data=SST("train",True)
    loader=DataLoader(train_data,batch_size=8,shuffle=True,num_workers=0,pin_memory=True,
                      generator=generator,drop_last=False)
    assert len(loader)==100
    model=WMSRPaper(scale=args.scale).cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,betas=(.9,.999),eps=1e-8)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=.5)
    frequency=FocalFrequencyLoss(loss_weight=1,alpha=1)
    start_epoch=1;step=0;prior_elapsed=0
    last_path=run/"checkpoints/last.pt"
    if last_path.exists():
        checkpoint=torch.load(last_path,map_location="cuda")
        assert checkpoint["config"]==config
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch=checkpoint["epoch"]+1;step=checkpoint["step"];prior_elapsed=checkpoint["elapsed_seconds"]
        random.setstate(checkpoint["rng_python"]);np.random.set_state(checkpoint["rng_numpy"])
        torch.set_rng_state(checkpoint["rng_torch"].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in checkpoint["rng_cuda"]])
        generator.set_state(checkpoint["rng_loader"].cpu())
    fields=["epoch","step","lr","loss","l1","ffl","epoch_seconds","elapsed_seconds"]
    started=time.monotonic();csv_path=run/"training_metrics.csv"
    with csv_path.open("a" if start_epoch>1 else "w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        if start_epoch==1:writer.writeheader()
        for epoch in range(start_epoch,args.epochs+1):
            model.train();begin=time.monotonic();sums=np.zeros(3);used_lr=optimizer.param_groups[0]["lr"]
            for lr,hr in loader:
                lr=lr.cuda(non_blocking=True);hr=hr.cuda(non_blocking=True)
                optimizer.zero_grad(set_to_none=True);sr=model(lr)
                l1=F.l1_loss(sr,hr);ffl=frequency(sr,hr);loss=.1*l1+.9*ffl
                if not torch.isfinite(loss):raise RuntimeError("Nonfinite loss at step"+str(step))
                loss.backward()
                if step%100==0:assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                optimizer.step();step+=1;sums+=np.array([loss.item(),l1.item(),ffl.item()])
            scheduler.step();elapsed=prior_elapsed+time.monotonic()-started
            row={"epoch":epoch,"step":step,"lr":used_lr,"loss":sums[0]/len(loader),"l1":sums[1]/len(loader),
                 "ffl":sums[2]/len(loader),"epoch_seconds":time.monotonic()-begin,"elapsed_seconds":elapsed}
            writer.writerow(row);stream.flush()
            checkpoint={"epoch":epoch,"step":step,"elapsed_seconds":elapsed,"model_state_dict":model.state_dict(),
                "optimizer_state_dict":optimizer.state_dict(),"scheduler_state_dict":scheduler.state_dict(),
                "config":config,"rng_python":random.getstate(),"rng_numpy":np.random.get_state(),
                "rng_torch":torch.get_rng_state(),"rng_cuda":torch.cuda.get_rng_state_all(),
                "rng_loader":generator.get_state()}
            temporary=last_path.with_suffix(".pt.tmp");torch.save(checkpoint,temporary);temporary.replace(last_path)
            json_save(run/"progress.json",{"status":"training","dataset":args.dataset,"scale":args.scale,
                       "epoch":epoch,"epochs":args.epochs,"step":step,"expected_steps":args.epochs*100,
                       "last_epoch":row,"pid":os.getpid()})
            print("EPOCH",args.dataset,args.scale,json.dumps(row),flush=True)
    assert step==args.epochs*100
    compact={"model_state_dict":model.state_dict(),"epoch":args.epochs,"step":step,"config":config}
    torch.save(compact,run/"checkpoints/final.pt")
    model.eval();test_data=SST("test")
    test=DataLoader(test_data,batch_size=8,shuffle=False,num_workers=0,pin_memory=True)
    @torch.inference_mode()
    def evaluate(predictor):
        rows=[]
        for lr,hr in test:
            lr=lr.cuda(non_blocking=True);hr=hr.cuda(non_blocking=True)
            sr=predictor(lr);border=args.scale
            physical=(sr.mean(1,keepdim=True)-hr.mean(1,keepdim=True))/255*42
            physical=physical[:,:,border:-border,border:-border]
            mse_c=physical.square().flatten(1).mean(1);mae_c=physical.abs().flatten(1).mean(1)
            y_sr=rgb_to_ycbcr(sr.clamp(0,255))[:,0:1,border:-border,border:-border]
            y_hr=rgb_to_ycbcr(hr.clamp(0,255))[:,0:1,border:-border,border:-border]
            mse_y=((y_sr.double()-y_hr.double())/255).square().flatten(1).mean(1)
            psnr=-10*torch.log10(mse_y.clamp_min(1e-16))
            structural=ssim(y_sr,y_hr,data_range=255,size_average=False)
            for values in zip(mse_c.tolist(),mae_c.tolist(),psnr.tolist(),structural.tolist()):
                rows.append({"mse_c2":values[0],"mae_c":values[1],"psnr_y_db":values[2],"ssim_y":values[3]})
        return summarize(rows),rows
    def summarize(rows):
        return {"count":len(rows),"rmse_c":math.sqrt(sum(x["mse_c2"] for x in rows)/len(rows)),
                "mae_c":sum(x["mae_c"] for x in rows)/len(rows),
                "psnr_y_db":sum(x["psnr_y_db"] for x in rows)/len(rows),
                "ssim_y":sum(x["ssim_y"] for x in rows)/len(rows)}
    wmsr,rows=evaluate(model)
    bicubic,baseline_rows=evaluate(lambda x:F.interpolate(x,scale_factor=args.scale,mode="bicubic",align_corners=False,antialias=False))
    for label,items in [("wmsr",rows),("bicubic",baseline_rows)]:
        with (run/(label+"_test_per_sample.csv")).open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=["sample_id","date","region_label","mse_c2","mae_c","psnr_y_db","ssim_y"]);writer.writeheader()
            for meta,item in zip(test_data.samples,items):
                writer.writerow({"sample_id":meta["sample_id"],"date":meta["date"],"region_label":meta.get("region_label",""),**item})
    regions={}
    if args.dataset=="HYCOM":
        for label in ["NP","AGM","NIO","EWP"]:
            selected=[i for i,meta in enumerate(test_data.samples) if meta["region_label"]==label]
            regions[label]={"wmsr":summarize([rows[i] for i in selected]),
                            "bicubic":summarize([baseline_rows[i] for i in selected])}
    result={"status":"complete","completed_at":dt.datetime.now(dt.timezone.utc).isoformat(),
            "config":config,"parameters":sum(p.numel() for p in model.parameters()),"epochs":args.epochs,
            "optimizer_steps":step,"test_wmsr":wmsr,"test_bicubic":bicubic,"per_region":regions,
            "delta_psnr_y_db":wmsr["psnr_y_db"]-bicubic["psnr_y_db"],
            "rmse_reduction_percent":100*(1-wmsr["rmse_c"]/bicubic["rmse_c"]),
            "elapsed_seconds":prior_elapsed+time.monotonic()-started,
            "gpu":torch.cuda.get_device_name(0),"torch":torch.__version__,"cuda_runtime":torch.version.cuda,
            "limits":config["protocol_limit"]}
    assert all(math.isfinite(float(value)) for metrics in [wmsr,bicubic] for value in metrics.values())
    json_save(run/"final_results.json",result);json_save(run/"progress.json",result)
    print("EXPERIMENT_COMPLETE",json.dumps(result),flush=True)
if __name__=="__main__":main()

