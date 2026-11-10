#!/usr/bin/env python3
"""Verify completed full-size runs and assemble a concise research record."""
import csv,datetime as dt,hashlib,json,math,subprocess,tarfile
from pathlib import Path
ROOT=Path("/root/wmsr-repro")
def main():
    subprocess.check_call([str(ROOT/".venv/bin/python"),str(ROOT/"scripts/15_verify_full_datasets.py")])
    subprocess.check_call([str(ROOT/".venv/bin/python"),str(ROOT/"scripts/17_plot_formal_examples.py")])
    jobs=[(name,scale) for name in ["HYCOM","OISST","GHRSST"] for scale in [2,3,4]]
    results=[]
    for name,scale in jobs:
        run=ROOT/"results/formal"/(name+"_x"+str(scale)+"_seed10")
        result=json.loads((run/"final_results.json").read_text())
        assert result["status"]=="complete" and result["epochs"]==100 and result["optimizer_steps"]==10000
        assert result["config"]["train_count"]==800 and result["config"]["test_count"]==200
        assert result["test_wmsr"]["count"]==200 and result["test_bicubic"]["count"]==200
        assert result["config"]["loss"]=={"L1":0.1,"FFL":0.9,"FFL_alpha":1}
        checkpoint=run/"checkpoints/final.pt";assert checkpoint.exists()
        result["checkpoint_sha256"]=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        results.append(result)
    out=ROOT/"results/summary";out.mkdir(exist_ok=True)
    fields=["dataset","scale","epochs","optimizer_steps","parameters","wmsr_psnr_y_db","bicubic_psnr_y_db",
            "delta_psnr_y_db","wmsr_ssim_y","bicubic_ssim_y","wmsr_rmse_c","bicubic_rmse_c",
            "wmsr_mae_c","bicubic_mae_c"]
    rows=[]
    for result in results:
        model=result["test_wmsr"];baseline=result["test_bicubic"];cfg=result["config"]
        rows.append({"dataset":cfg["dataset"],"scale":cfg["scale"],"epochs":result["epochs"],
                     "optimizer_steps":result["optimizer_steps"],"parameters":result["parameters"],
                     "wmsr_psnr_y_db":model["psnr_y_db"],"bicubic_psnr_y_db":baseline["psnr_y_db"],
                     "delta_psnr_y_db":result["delta_psnr_y_db"],"wmsr_ssim_y":model["ssim_y"],
                     "bicubic_ssim_y":baseline["ssim_y"],"wmsr_rmse_c":model["rmse_c"],
                     "bicubic_rmse_c":baseline["rmse_c"],"wmsr_mae_c":model["mae_c"],"bicubic_mae_c":baseline["mae_c"]})
    with (out/"metrics.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    timezone=dt.timezone(dt.timedelta(hours=8));now=dt.datetime.now(timezone).isoformat()
    lines=["# WMSR 公开材料近似重建：全规模实验结果","",
           "统计时间："+now,"",
           "本轮已完成三个数据集、三个倍率共9组训练。每组800张训练图、200张测试图，100epochs、10000次参数更新、单一种子10。",
           "这是基于公开原始SST数据与论文结构解释的重建，不是已验证的作者私有实验版本；不能将分数接近或更高直接称为复现成功。","",
           "## 实测测试结果","",
           "| 数据 | 倍率 | 模型PSNR(Y) | Bicubic PSNR(Y) | 差值dB | 模型SSIM(Y) | 模型RMSE °C | Bicubic RMSE °C |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| "+row["dataset"]+" | ×"+str(row["scale"])+" | "+format(row["wmsr_psnr_y_db"],".4f")+" | "+format(row["bicubic_psnr_y_db"],".4f")+" | "+format(row["delta_psnr_y_db"],"+.4f")+" | "+format(row["wmsr_ssim_y"],".6f")+" | "+format(row["wmsr_rmse_c"],".6f")+" | "+format(row["bicubic_rmse_c"],".6f")+" |")
    wins=sum(row["delta_psnr_y_db"]>0 for row in rows)
    lines.extend(["","按本轮统一协议，有"+str(wins)+"/9组的PSNR(Y)高于Bicubic。训练与评测流程完成不等于达到论文性能；无论正负差值均完整保留。"])
    control_path=ROOT/"results/public_control/OISST_x4_seed10/final_results.json"
    if control_path.exists():
        control=json.loads(control_path.read_text())
        assert control["epochs"]==100 and control["optimizer_steps"]==10000 and control["test_wmsr"]["count"]==200
        lines.extend(["","## 作者公开网络的对照","",
            "另在相同OISST800/200划分和100epochs训练设置下运行了未修改的作者公开网络（实际4个WAM），用于区分公开源码对象与按图重建对象；初始化依各自实现，不作纯结构优劣结论。",
            "公开网络PSNR(Y)="+format(control["test_wmsr"]["psnr_y_db"],".4f")+"dB，SSIM(Y)="+format(control["test_wmsr"]["ssim_y"],".6f")+"，温度RMSE="+format(control["test_wmsr"]["rmse_c"],".6f")+"°C。",
            "对应配置、日志摘要和权重位于 ../public_control/OISST_x4_seed10/，不混入主表9组重建实验。"])
    lines+=["","## 训练和指标口径","",
            "- 结构：6个RWAMG，每组4个WAM；外层64通道、内部32通道瓶颈是未公开尺寸的重建假设。×4实际963848参数，原文约977K；参数接近不是实现一致的证明。",
            "- Adam，初始lr=1e-3，beta=(0.9,0.999)、epsilon=1e-8；每20epochs学习率减半。",
            "- 损失=0.1×L1+0.9×FocalFrequencyLoss(alpha=1)。",
            "- 每个epoch完整遍历800张训练图一次，batch8，repeat1，共10000次更新；论文未说明repeat，公开默认配置为repeat20，这是需要明确的训练协议解释。",
            "- 96×96采用公开loader的HR patch定义，对应LR48/32/24。原文input patch措辞不足以确认HR/LR含义。",
            "- 原始温度保持float32 Celsius，固定[-2,40]线性映射至0..255并重复为三通道；作者RGB/伪彩色映射未公开，不能确认与之相同。",
            "- LR为PyTorch bicubic、antialias=True、align_corners=False；测试Bicubic回放antialias=False。",
            "- 图像指标按公开代码方式：clamp到0..255、RGB转BT.601 Y、裁剪倍率像素边界，逐图PSNR/SSIM平均；保留无偏的完整精度。",
            "- 温度RMSE/MAE由未裁值预测的三通道均值反归一化得到；不与原文未报告的温度误差混为一谈。",
            "- 使用最终第100epoch权重，未按测试集选择checkpoint。没有额外从800张训练图中扣除验证集。","",
            "## 数据完整性与来源",""]
    for name in ["HYCOM","OISST","GHRSST"]:
        manifest=json.loads((ROOT/"records/formal"/(name+"_manifest.json")).read_text())
        lines.append("- "+name+"：1000张不同ID，800/200；来源及每图裁剪记录见 ../../records/formal/"+name+"_manifest.json。")
        if name=="OISST":
            lines.append("  NOAA OISST v2.1，2016年366个原始日文件；按全球海洋采样，训练/测试来源日期隔离。")
        elif name=="GHRSST":
            lines.append("  MUR v4.1，2016-08-12，原生0.01°；全球非极地海洋的1000块原生图像，训练/测试空间不重叠。产品选择和裁剪清单并非作者提供。")
        else:
            lines.append("  GOFS3.1 GLBv0.08 expt56.3/57.2，2016年原生3小时产品、0m温度。每区域250个不同时间点，200训练/50测试；原文四个区域的数值经纬度按字面保留。")
    lines+=["","## 与原文的核对边界","",
            "论文结构组数的文字、图表存在不一致；此处采用Fig.1和TableII的6×4。频率损失权重按Fig.4的0.1/0.9。学习率同时出现cosine与每20epoch减半，采用后者。",
            "HYCOM采用原文数值坐标；AGM/NIO/EWP名称与坐标的地理意义并不协调，不会将标签当成已验证的地名。",
            "缺少作者原始HR文件清单、划分、预处理和匹配权重；本轮只能说明这个明确记录的重建协议在单种子下的表现，不能建立作者原方法的精确数值结论，也不能外推真实卫星退化或其他年份泛化。",
            "训练在共享单卡上并行执行独立实验；逐组墙钟耗时不是可与论文独占GPU实验直接对比的速度指标。","",
            "## 可复查文件","",
            "- 完整结果表：[metrics.csv](./metrics.csv)",
            "- 训练曲线：[training_curves.png](./training_curves.png)",
            "- 固定测试序号0/66/133的温度与误差图：examples_数据集_x倍率.png；选择清单见 example_selection.json，未按测试分数挑图。图中温度是三输出通道均值的反归一化值。",
            "- 每组配置、逐图测试分数、最终权重和源码副本位于 ../formal/。",
            "- 原始数据URL/哈希、环境锁定和模型假设位于 ../../records/。",
            "- 各组last.pt（含优化器/RNG）保留在云端用于恢复；本地归档只包含关键final.pt，不复制数据、环境或缓存。","",
            "来源：[论文](https://arxiv.org/abs/2509.24334)、[作者源码](https://github.com/oucailab/WMSR)、[NOAA OISST](https://www.ncei.noaa.gov/products/optimum-interpolation-sst)、[MUR产品](https://podaac.jpl.nasa.gov/dataset/MUR-JPL-L4-GLOB-v4.1)。",""]
    (out/"README.md").write_text("\n".join(lines))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12.6,3.7),layout="constrained")
    colors={"HYCOM":"#406a98","OISST":"#39816f","GHRSST":"#b77749"}
    for ax,scale in zip(axes,[2,3,4]):
        for name in ["HYCOM","OISST","GHRSST"]:
            with (ROOT/"results/formal"/f"{name}_x{scale}_seed10/training_metrics.csv").open() as stream:data=list(csv.DictReader(stream))
            ax.plot([int(x["epoch"]) for x in data],[float(x["loss"]) for x in data],label=name,color=colors[name])
        ax.set_title("Scale x"+str(scale));ax.set_xlabel("Epoch");ax.set_yscale("log");ax.grid(alpha=.2)
    axes[0].set_ylabel("0.1 L1 + 0.9 FFL");axes[-1].legend()
    fig.suptitle("Full-size public reconstruction / seed10 / all100epochs")
    fig.savefig(out/"training_curves.png",dpi=170);plt.close(fig)
    verification={"checked_at":now,"jobs":9,"all_epochs100":True,"all_optimizer_steps10000":True,
                  "all_train800_test200":True,"all_final_checkpoints_present":True,
                  "checkpoints":[{"dataset":x["config"]["dataset"],"scale":x["config"]["scale"],
                                  "sha256":x["checkpoint_sha256"]} for x in results],
                  "reproduction_boundary":"Explicit approximate reconstruction; private author protocol unavailable"}
    (out/"verification.json").write_text(json.dumps(verification,indent=2)+"\n")
    allowed=[]
    upstream_commit=subprocess.check_output(["git","-C",str(ROOT/"upstream"),"rev-parse","HEAD"],text=True).strip()
    assert upstream_commit=="dbb1cff4c4b30375928c6ee62224cb4b37fa337a"
    upstream_archive=ROOT/"records"/("WMSR_upstream_"+upstream_commit[:12]+".tar.gz")
    subprocess.check_call(["git","-C",str(ROOT/"upstream"),"archive","--format=tar.gz",
                           "--output="+str(upstream_archive),upstream_commit])
    for pattern in ["scripts/*.py","scripts/*.sh","paper_models/*.py","configs/*.txt","records/*.json","records/*.md",
                    "records/environment.lock.txt","records/WMSR_upstream_*.tar.gz","records/formal/*.json","results/summary/*",
                    "results/formal/*/config.json","results/formal/*/model_source.py","results/formal/*/final_results.json",
                    "results/formal/*/training_metrics.csv","results/formal/*/*test_per_sample.csv","results/formal/*/checkpoints/final.pt"]:
        allowed.extend(p for p in ROOT.glob(pattern) if p.is_file() and not p.is_symlink())
    for pattern in ["results/public_control/*/config.json","results/public_control/*/model_source.py",
                    "results/public_control/*/final_results.json","results/public_control/*/training_metrics.csv",
                    "results/public_control/*/*test_per_sample.csv","results/public_control/*/checkpoints/final.pt"]:
        allowed.extend(p for p in ROOT.glob(pattern) if p.is_file() and not p.is_symlink())
    bundle=ROOT/"results/WMSR_important_results.tar.gz"
    with tarfile.open(bundle,"w:gz",compresslevel=5) as archive:
        for path in sorted(set(allowed)):archive.add(path,arcname=str(path.relative_to(ROOT)),recursive=False)
    digest=hashlib.sha256(bundle.read_bytes()).hexdigest()
    (ROOT/"results/WMSR_important_results.sha256").write_text(digest+"  "+bundle.name+"\n")
    (ROOT/"records/results_package.json").write_text(json.dumps({"status":"complete","path":str(bundle),
          "bytes":bundle.stat().st_size,"sha256":digest,"packaged_files":len(set(allowed))},indent=2)+"\n")
    print("RESULTS_VERIFIED_AND_PACKAGED",bundle,bundle.stat().st_size,digest,flush=True)
if __name__=="__main__":main()

