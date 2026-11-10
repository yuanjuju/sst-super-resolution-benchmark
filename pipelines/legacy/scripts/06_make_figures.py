#!/usr/bin/env python3
"""Make scientific figures from completed cloud results, using the fixed first test crop."""
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

def main():
    root=Path("/root/wmsr-repro")
    run=root/"results/oisst_x4_seed10"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    result=json.loads((run/"final_results.json").read_text())
    with (run/"training_metrics.csv").open() as stream:rows=list(csv.DictReader(stream))
    val=[row for row in rows if row["val_rmse_c"]]
    baseline=json.loads((run/"bicubic_validation.json").read_text())
    fig,axes=plt.subplots(1,3,figsize=(12.8,3.6),layout="constrained")
    axes[0].plot([int(row["epoch"]) for row in rows],[float(row["train_loss"]) for row in rows],color="#315f92")
    axes[0].set_yscale("log");axes[0].set_ylabel("0.9 L1 + 0.1 FFL");axes[0].set_title("Training loss")
    axes[1].plot([int(row["epoch"]) for row in val],[float(row["val_rmse_c"]) for row in val],marker="o",markersize=3,color="#315f92",label="WMSR")
    axes[1].axhline(baseline["rmse_c"],color="#9c593c",ls="--",label="Bicubic")
    axes[1].set_ylabel("RMSE (degrees Celsius)");axes[1].set_title("Validation temperature error");axes[1].legend()
    axes[2].plot([int(row["epoch"]) for row in val],[float(row["val_psnr_db"]) for row in val],marker="o",markersize=3,color="#497f68")
    axes[2].axhline(baseline["psnr_db"],color="#9c593c",ls="--")
    axes[2].set_ylabel("PSNR (dB; fixed 42 C range)");axes[2].set_title("Validation PSNR")
    for ax in axes:ax.set_xlabel("Epoch");ax.grid(alpha=0.2)
    fig.suptitle("Public WMSR / reconstructed OISST x4 / seed 10",fontsize=12)
    fig.savefig(run/"training_curves.png",dpi=160);plt.close(fig)

    sys.path.insert(0,str(root/"upstream"))
    from models.wmsr_network import create_model
    model=create_model(SimpleNamespace(**result["config"])).cuda().eval()
    checkpoint=torch.load(run/"checkpoints/best.pt",map_location="cuda")
    model.load_state_dict(checkpoint["model_state_dict"])
    with np.load(root/"data/oisst_np_x4/test.npz") as arrays:
        hr=arrays["hr_celsius"][0].copy();lr=arrays["lr_celsius"][0].copy()
    tensor=torch.from_numpy((lr+2.0)/42.0*255.0)[None,None].repeat(1,3,1,1).cuda()
    with torch.inference_mode():
        sr=(model(tensor).mean(1)[0]/255.0*42.0-2.0).cpu().numpy()
        bicubic=F.interpolate(torch.from_numpy(lr)[None,None],scale_factor=4,mode="bicubic",align_corners=False,antialias=False)[0,0].numpy()
    with (run/"wmsr_test_per_sample.csv").open() as stream:sample_wmsr=next(csv.DictReader(stream))
    with (run/"bicubic_test_per_sample.csv").open() as stream:sample_bicubic=next(csv.DictReader(stream))
    titles=["HR temperature (96 x 96)","LR temperature (24 x 24)",
            "Bicubic / RMSE "+format(float(sample_bicubic["mse_c2"])**0.5,".3f")+" C",
            "WMSR / RMSE "+format(float(sample_wmsr["mse_c2"])**0.5,".3f")+" C"]
    fig,axes=plt.subplots(1,4,figsize=(14,3.7),layout="constrained")
    for ax,array,title in zip(axes,[hr,lr,bicubic,sr],titles):
        im=ax.imshow(array,origin="lower",cmap="viridis",vmin=float(hr.min()),vmax=float(hr.max()),interpolation="nearest")
        ax.set_title(title,fontsize=10);ax.set_xticks([]);ax.set_yticks([])
    fig.colorbar(im,ax=axes,label="SST (degrees Celsius)",shrink=0.82)
    fig.suptitle("First fixed test sample: "+sample_wmsr["sample_id"]+" / metrics exclude 4-pixel border",fontsize=11)
    fig.savefig(run/"comparison_first_test.png",dpi=160);plt.close(fig)
    print("FIGURES_COMPLETE",flush=True)

if __name__=="__main__":
    main()

