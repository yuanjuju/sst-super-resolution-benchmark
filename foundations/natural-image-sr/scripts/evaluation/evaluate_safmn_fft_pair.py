#!/usr/bin/env python3
"""Evaluate the completed, single-seed SAFMN FFT/no-FFT 30k experiment.

Run with the SAFMN project environment. No dependencies are installed and no
training is started. All paths default to the cloud project root.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from datetime import datetime, timezone


STEPS = (5000, 10000, 15000, 20000, 25000, 30000)
EVAL_STEPS = (5000, 30000)
ARMS = ("fft", "nofft")
PROTOCOL = "BasicSR calculate_psnr/calculate_ssim; uint8 BGR; RGB and Y; crop_border=4; params_ema"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No records for {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_training_metrics(log_path, arm):
    """Associate validation with actual progress, never the logger's Best iter.

    BasicSR can validate again after its end marker (and report total_iter+2).
    Such final duplicate validation is intentionally excluded.
    """
    latest_iter = None
    pending_validation = None
    ended = False
    records = {}
    progress_re = re.compile(r"\[epoch:\s*\d+,\s*iter:\s*([\d,]+),")
    metric_re = re.compile(r"#\s*psnr:\s*([-+\d.eE]+)(?:\s|$)")
    for line_number, line in enumerate(Path(log_path).read_text(errors="replace").splitlines(), 1):
        if "End of training" in line:
            ended = True
            pending_validation = None
        if ended:
            continue
        progress = progress_re.search(line)
        if progress:
            latest_iter = int(progress.group(1).replace(",", ""))
        if "Validation DIV2K_val100" in line:
            if latest_iter not in STEPS:
                raise ValueError(f"{log_path}:{line_number}: validation without an expected actual progress step")
            pending_validation = (latest_iter, line_number)
            continue
        metric = metric_re.search(line)
        if metric and pending_validation:
            step, validation_line = pending_validation
            value = float(metric.group(1))
            if not math.isfinite(value):
                raise ValueError(f"Non-finite validation PSNR in {log_path}:{line_number}")
            if step in records:
                raise ValueError(f"Duplicate pre-end validation at {step} in {log_path}")
            records[step] = {
                "arm": arm, "iter": step, "psnr_rgb_db": value,
                "dataset": "DIV2K_val100", "weight_key": "params_ema",
                "crop_border": 4, "test_y_channel": False,
                "source_log": str(log_path), "validation_line": validation_line,
                "metric_line": line_number,
            }
            pending_validation = None
    if not ended:
        raise ValueError(f"Training completion marker missing: {log_path}")
    if set(records) != set(STEPS):
        raise ValueError(f"Expected validations {STEPS}; got {tuple(sorted(records))} in {log_path}")
    return [records[step] for step in STEPS]


def pillow_modules():
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageOps
        return Image, ImageDraw, ImageFont, ImageOps
    except ImportError:
        return None


def pillow_font(ImageFont, size):
    for filename in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(filename, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_comparison(path, panels, stem, pil):
    """All methods use the same full-image preview and center ROI coordinates."""
    Image, ImageDraw, ImageFont, ImageOps = pil
    width, height = panels[0][1].size
    if any(image.size != (width, height) for _, image in panels):
        raise ValueError("Comparison panels have different dimensions")
    roi_w, roi_h = max(1, width // 4), max(1, height // 4)
    left, top = (width - roi_w) // 2, (height - roi_h) // 2
    roi = (left, top, left + roi_w, top + roi_h)
    cell, gap, full_top, roi_top = 256, 16, 100, 400
    canvas = Image.new("RGB", (gap + len(panels) * (cell + gap), 692), "#f6f7fb")
    draw = ImageDraw.Draw(canvas)
    title_font, font, small = (pillow_font(ImageFont, size) for size in (21, 15, 13))
    draw.text((gap, 12), f"SAFMN x4 | Set5/{stem} | single-seed FFT loss comparison", fill="#172033", font=title_font)
    draw.text((gap, 43), "EMA weights; displayed PSNR is RGB, crop 4. Shared center ROI is marked in red.", fill="#475569", font=font)
    for column, (label, image) in enumerate(panels):
        x = gap + column * (cell + gap)
        draw.text((x, 76), label, fill="#172033", font=font)
        view = ImageOps.contain(image, (cell, cell), Image.Resampling.LANCZOS)
        px, py = x + (cell - view.width) // 2, full_top + (cell - view.height) // 2
        canvas.paste(view, (px, py))
        draw.rectangle((px + left * view.width / width, py + top * view.height / height,
                        px + roi[2] * view.width / width, py + roi[3] * view.height / height),
                       outline="#dc2626", width=2)
        detail = ImageOps.contain(image.crop(roi), (cell, cell), Image.Resampling.NEAREST)
        canvas.paste(detail, (x + (cell - detail.width) // 2, roi_top + (cell - detail.height) // 2))
    draw.text((gap, 371), "Shared ROI enlarged with nearest-neighbor display; enlargement adds no information.", fill="#475569", font=font)
    draw.text((gap, 670), "Pillow bicubic is a display baseline, not the official MATLAB degradation generator. No model ranking or full-training claim.", fill="#475569", font=small)
    canvas.save(path)
    return list(roi)


def draw_curve(path, records, pil):
    """Use matplotlib if present; otherwise draw a labeled educational curve."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
    if plt is not None:
        fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
        try:
            for arm, color in (("fft", "#2563eb"), ("nofft", "#d97706")):
                points = [row for row in records if row["arm"] == arm]
                ax.plot([p["iter"] for p in points], [p["psnr_rgb_db"] for p in points],
                        "o-", color=color, label="With FFT" if arm == "fft" else "Without FFT")
            ax.set(title="SAFMN x4: single-seed FFT loss comparison",
                   xlabel="Training iteration", ylabel="DIV2K_val100 RGB PSNR (dB), EMA; crop 4")
            ax.set_xticks(STEPS, [f"{step // 1000}k" for step in STEPS])
            ax.grid(alpha=0.25)
            ax.legend()
            fig.savefig(path, dpi=180)
        finally:
            plt.close(fig)
        return "matplotlib"
    if pil is None:
        return None
    Image, ImageDraw, ImageFont, _ = pil
    image = Image.new("RGB", (1100, 670), "white")
    draw = ImageDraw.Draw(image)
    font, small = pillow_font(ImageFont, 20), pillow_font(ImageFont, 16)
    draw.text((45, 20), "SAFMN x4: single-seed FFT loss comparison", fill="#172033", font=font)
    draw.text((45, 55), "DIV2K_val100 RGB PSNR (dB); EMA; crop border 4", fill="#475569", font=small)
    x0, y0, x1, y1 = 100, 110, 1030, 555
    values = [row["psnr_rgb_db"] for row in records]
    low, high = math.floor((min(values) - 0.15) * 2) / 2, math.ceil((max(values) + 0.15) * 2) / 2
    def position(step, score):
        return (x0 + (step - STEPS[0]) / (STEPS[-1] - STEPS[0]) * (x1 - x0),
                y1 - (score - low) / (high - low) * (y1 - y0))
    for index in range(6):
        value = low + (high - low) * index / 5
        y = position(STEPS[0], value)[1]
        draw.line((x0, y, x1, y), fill="#e2e8f0", width=1)
        draw.text((20, y - 9), f"{value:.2f}", fill="#475569", font=small)
    draw.line((x0, y0, x0, y1, x1, y1), fill="#334155", width=2)
    for step in STEPS:
        x = position(step, low)[0]
        draw.text((x - 16, y1 + 12), f"{step // 1000}k", fill="#475569", font=small)
    for arm, color, offset in (("fft", "#2563eb", -23), ("nofft", "#d97706", 9)):
        points = [row for row in records if row["arm"] == arm]
        xy = [position(row["iter"], row["psnr_rgb_db"]) for row in points]
        draw.line(xy, fill=color, width=3)
        for (x, y), row in zip(xy, points):
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
            draw.text((x - 25, y + offset), f"{row['psnr_rgb_db']:.3f}", fill=color, font=small)
    draw.text((100, 605), "Blue: with FFT     Orange: without FFT     x-axis: training iteration", fill="#334155", font=small)
    draw.text((100, 635), "Six validation checkpoints per run; no interpolation-based convergence claim.", fill="#475569", font=small)
    image.save(path)
    return "Pillow educational line chart"


def run(args):
    root = Path(args.root).expanduser().resolve()
    suite = (Path(args.suite).expanduser() if args.suite else root / "safmn_fft_pair_30k").resolve()
    checkout = root / "SAFMN"
    results = suite / "results"
    results.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(suite / ".cache" / "matplotlib"))
    record = {"completed": False, "experiment": "SAFMN x4 single-seed FFT/no-FFT 30k",
              "root": str(root), "suite": str(suite), "protocol": PROTOCOL,
              "artifacts": {}, "warnings": []}
    write_json(results / "evaluation.json", record)
    try:
        logs = {arm: suite / f"{arm}.stdout.log" for arm in ARMS}
        training = [row for arm in ARMS for row in parse_training_metrics(logs[arm], arm)]
        training_csv = results / "training_metrics.csv"
        write_csv(training_csv, training)
        sys.path.insert(0, str(checkout))
        import cv2
        import numpy as np
        import torch
        import basicsr
        from basicsr.archs import build_network
        from basicsr.metrics import calculate_psnr, calculate_ssim
        from basicsr.utils import tensor2img
        module_path = Path(basicsr.__file__).resolve()
        if not module_path.is_relative_to(checkout):
            raise ValueError(f"Expected the SAFMN checkout BasicSR, got {module_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pil = pillow_modules()
        if pil is None:
            record["warnings"].append("Pillow unavailable: comparison figures and bicubic baseline skipped.")
        hr_dir = root / "datasets" / "Set5" / "HR"
        lq_dir = root / "datasets" / "Set5" / "LR_bicubic" / "X4"
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        hr_paths = sorted(path for path in hr_dir.iterdir() if path.suffix.lower() in extensions)
        if len(hr_paths) != 5:
            raise ValueError(f"Expected exactly 5 Set5 HR images, found {len(hr_paths)}")
        samples, sources = [], {}
        for hr_path in hr_paths:
            lq_path = lq_dir / hr_path.name
            lq, gt = cv2.imread(str(lq_path), cv2.IMREAD_COLOR), cv2.imread(str(hr_path), cv2.IMREAD_COLOR)
            if lq is None or gt is None:
                raise ValueError(f"Unreadable Set5 pair: {lq_path}, {hr_path}")
            if min(lq.shape[:2]) < 8 or gt.shape[:2] != (lq.shape[0] * 4, lq.shape[1] * 4):
                raise ValueError(f"Invalid x4 image dimensions for {hr_path.name}")
            sample = {"name": hr_path.stem, "lq": lq, "gt": gt, "predictions": {}}
            samples.append(sample)
            for source in (lq_path, hr_path):
                sources[str(source)] = sha256(source)
        network_options = {"type": "SAFMN", "dim": 36, "n_blocks": 8,
                           "ffn_scale": 2.0, "upscaling_factor": 4}
        model = build_network(network_options.copy()).to(device).eval()
        per_image, checkpoint_info = [], []
        def measure(prediction, gt):
            values = {}
            for label, y_channel in (("rgb", False), ("y", True)):
                values[f"psnr_{label}_db"] = float(calculate_psnr(prediction, gt, crop_border=4,
                                                               input_order="HWC", test_y_channel=y_channel))
                values[f"ssim_{label}"] = float(calculate_ssim(prediction, gt, crop_border=4,
                                                             input_order="HWC", test_y_channel=y_channel))
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError("Non-finite Set5 metric")
            return values
        for arm in ARMS:
            exp = checkout / "experiments" / f"SAFMN_DF2K_x4_pair_{arm}_30000"
            for step in EVAL_STEPS:
                weights_path = exp / "models" / f"net_g_{step}.pth"
                checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
                if not isinstance(checkpoint, dict) or "params_ema" not in checkpoint:
                    raise ValueError(f"Checkpoint has no params_ema: {weights_path}")
                weights = checkpoint["params_ema"]
                if any(not torch.isfinite(value).all().item() for value in weights.values()):
                    raise ValueError(f"Non-finite checkpoint tensor: {weights_path}")
                model.load_state_dict(weights, strict=True)
                checkpoint_info.append({"arm": arm, "iter": step, "path": str(weights_path),
                                        "sha256": sha256(weights_path), "weight_key": "params_ema",
                                        "checkpoint_keys": list(checkpoint)})
                del checkpoint, weights
                for sample in samples:
                    rgb = np.ascontiguousarray(sample["lq"][..., ::-1].transpose(2, 0, 1))
                    tensor = torch.from_numpy(rgb).float().div_(255).unsqueeze(0).to(device)
                    with torch.inference_mode():
                        output = model(tensor)
                    if not torch.isfinite(output).all().item():
                        raise ValueError(f"Non-finite output: {arm}, {step}, {sample['name']}")
                    prediction = tensor2img(output, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1))
                    if prediction.shape != sample["gt"].shape:
                        raise ValueError(f"Prediction/GT shape mismatch: {sample['name']}")
                    prediction_path = results / "predictions" / sample["name"] / f"{arm}_{step}.png"
                    prediction_path.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(prediction_path), prediction):
                        raise OSError(f"Failed to save {prediction_path}")
                    metrics = measure(prediction, sample["gt"])
                    row = {"arm": arm, "iter": step, "image": sample["name"],
                           "weight_key": "params_ema", "crop_border": 4, **metrics,
                           "input_width": sample["lq"].shape[1], "input_height": sample["lq"].shape[0],
                           "output_width": prediction.shape[1], "output_height": prediction.shape[0],
                           "prediction_path": str(prediction_path)}
                    per_image.append(row)
                    sample["predictions"][(arm, step)] = (prediction, metrics)
                    print(f"{arm} {step}: {sample['name']} RGB PSNR={metrics['psnr_rgb_db']:.4f}, Y PSNR={metrics['psnr_y_db']:.4f}", flush=True)
                    del output, tensor
        metric_keys = ("psnr_rgb_db", "ssim_rgb", "psnr_y_db", "ssim_y")
        means = []
        for arm in ARMS:
            for step in EVAL_STEPS:
                rows = [row for row in per_image if row["arm"] == arm and row["iter"] == step]
                if len(rows) != 5:
                    raise ValueError("Incomplete five-image Set5 evaluation")
                means.append({"arm": arm, "iter": step, "images": 5, "weight_key": "params_ema",
                              "crop_border": 4,
                              **{key: sum(row[key] for row in rows) / len(rows) for key in metric_keys}})
        write_csv(results / "set5_per_image.csv", per_image)
        write_csv(results / "set5_mean.csv", means)
        write_json(results / "set5_metrics.json", {"protocol": PROTOCOL,
                   "mean_definition": "Arithmetic mean of five per-image metrics, not pooled pixel MSE",
                   "per_image": per_image, "means": means})
        rois = {}
        if pil:
            Image = pil[0]
            figure_dir = results / "comparisons"
            figure_dir.mkdir(exist_ok=True)
            for sample in samples:
                def image_from_bgr(array):
                    return Image.fromarray(np.ascontiguousarray(array[..., ::-1]))
                gt_image = image_from_bgr(sample["gt"])
                bicubic = image_from_bgr(sample["lq"]).resize(gt_image.size, Image.Resampling.BICUBIC)
                panels = [("Pillow bicubic", bicubic)]
                for arm, step in (("fft", 5000), ("nofft", 5000), ("fft", 30000), ("nofft", 30000)):
                    prediction, metrics = sample["predictions"][(arm, step)]
                    label = f"{'FFT' if arm == 'fft' else 'noFFT'} {step // 1000}k | {metrics['psnr_rgb_db']:.3f} dB"
                    panels.append((label, image_from_bgr(prediction)))
                panels.append(("HR ground truth", gt_image))
                rois[sample["name"]] = draw_comparison(figure_dir / f"{sample['name']}.png", panels, sample["name"], pil)
        curve_path = results / "training_curve.png"
        renderer = draw_curve(curve_path, training, pil)
        if renderer is None:
            record["warnings"].append("No matplotlib or Pillow: training_curve.png could not be generated.")
        architecture_path = checkout / "basicsr" / "archs" / "safmn_arch.py"
        sources[str(architecture_path)] = sha256(architecture_path)
        for log_path in logs.values():
            sources[str(log_path)] = sha256(log_path)
        provenance_paths = [suite / "configs" / f"train_{arm}_30000.yml" for arm in ARMS]
        provenance_paths.extend((suite / "environment.txt", suite / "common_init.pth",
                                 suite / "preflight.json", Path(__file__).resolve()))
        for source in provenance_paths:
            if source.is_file():
                sources[str(source)] = sha256(source)
        record.update({"completed": True, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                       "device": str(device), "torch_version": str(torch.__version__),
                       "cuda_runtime": torch.version.cuda,
                       "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                       "numpy_version": np.__version__, "opencv_version": cv2.__version__,
                       "basicsr_path": str(module_path),
                       "network": network_options, "parameter_count": sum(p.numel() for p in model.parameters()),
                       "checkpoints": checkpoint_info, "source_sha256": sources,
                       "training_metrics": training, "set5_means": means,
                       "comparison_center_roi_xyxy": rois, "curve_renderer": renderer,
                       "notes": ["Single-seed controlled short-training comparison; not multi-seed evidence or full paper reproduction.",
                                 "Training curve excludes validation emitted after End of training.",
                                 "No inference speed benchmark is performed.",
                                 "Pillow bicubic is a visualization baseline, not the official MATLAB degradation generator."]})
        artifact_paths = [training_csv, results / "set5_per_image.csv", results / "set5_mean.csv", results / "set5_metrics.json"]
        artifact_paths.extend(sorted((results / "predictions").rglob("*.png")))
        if pil:
            artifact_paths.extend(sorted((results / "comparisons").glob("*.png")))
        if renderer:
            artifact_paths.append(curve_path)
        record["artifacts"] = {str(path.relative_to(results)): {"path": str(path), "sha256": sha256(path)}
                               for path in artifact_paths}
        write_json(results / "evaluation.json", record)
        print(f"Evaluation completed. Records: {results / 'evaluation.json'}", flush=True)
    except Exception as error:
        record.update({"completed": False, "error_type": type(error).__name__, "error": str(error)})
        write_json(results / "evaluation.json", record)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/root/fsas/sr-repro", help="Cloud project root (default: %(default)s)")
    parser.add_argument("--suite", help="Experiment suite directory (default: ROOT/safmn_fft_pair_30k)")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
