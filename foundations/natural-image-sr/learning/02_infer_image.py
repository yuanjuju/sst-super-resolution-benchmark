#!/usr/bin/env python3
"""Run the locally trained SAFMN checkpoint on one RGB image, using CPU only.

Relative CLI paths are resolved from the repository root, independently of cwd.
The default Set5 example includes a ground-truth image; custom inputs do not.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch

from _safmn_source import load_safmn


REPO = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO / "datasets/Set5/LR_bicubic/X4/butterfly.png"
DEFAULT_GT = REPO / "datasets/Set5/HR/butterfly.png"
DEFAULT_WEIGHTS = REPO / "downloads/net_g_30000.pth"


def repo_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO / path).resolve()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def rgb_psnr(image, target, crop=4):
    """Measure saved uint8 RGB images, with float64 MSE and a 255 peak."""
    if image.size != target.size:
        raise ValueError(f"GT size {target.size} does not match output {image.size}.")
    a = np.asarray(image, dtype=np.uint8)
    b = np.asarray(target, dtype=np.uint8)
    if min(a.shape[:2]) <= 2 * crop:
        raise ValueError(f"Images must be larger than {2 * crop} pixels for crop {crop}.")
    a = a[crop:-crop, crop:-crop].astype(np.float64)
    b = b[crop:-crop, crop:-crop].astype(np.float64)
    mse = float(np.mean((a - b) ** 2, dtype=np.float64))
    return {"psnr_db": "Infinity" if mse == 0 else 10 * math.log10(255 ** 2 / mse),
            "mse": mse}


def checkpoint_label(path):
    match = re.fullmatch(r"net_g_(\d+)\.pth", path.name)
    if not match:
        return "SAFMN"
    step = int(match.group(1))
    return f"SAFMN {step // 1000}k" if step % 1000 == 0 else f"SAFMN step {step}"


def comparison_image(panels):
    """Show identical full-image and center-ROI locations for all methods."""
    width, height = panels[0][1].size
    roi_w, roi_h = max(1, width // 4), max(1, height // 4)
    left, top = (width - roi_w) // 2, (height - roi_h) // 2
    roi = (left, top, left + roi_w, top + roi_h)
    cell_w, cell_h, gap = 320, 320, 16
    roi_h_display = 160
    canvas_w = gap + len(panels) * (cell_w + gap)
    canvas = Image.new("RGB", (canvas_w, 660), "#f7f8fa")
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.load_default(size=20)
    label_font = ImageFont.load_default(size=14)
    small_font = ImageFont.load_default(size=12)
    draw.text((gap, 12), "SAFMN single-image CPU inference | x4", fill="#172033", font=title_font)
    draw.text((gap, 42), "Full image; red box marks the shared center ROI", fill="#4b5563", font=label_font)
    for column, (label, image) in enumerate(panels):
        x = gap + column * (cell_w + gap)
        draw.text((x, 69), label, fill="#172033", font=label_font)
        # Keep small images at their native display size; downsize large previews.
        view = image.copy()
        if view.width > cell_w or view.height > cell_h:
            view = ImageOps.contain(view, (cell_w, cell_h), Image.Resampling.LANCZOS)
        px, py = x + (cell_w - view.width) // 2, 94 + (cell_h - view.height) // 2
        canvas.paste(view, (px, py))
        sx, sy = view.width / width, view.height / height
        draw.rectangle((px + left * sx, py + top * sy,
                        px + (left + roi_w) * sx, py + (top + roi_h) * sy),
                       outline="#dc2626", width=2)
        detail = ImageOps.contain(image.crop(roi), (cell_w, roi_h_display), Image.Resampling.NEAREST)
        canvas.paste(detail, (x + (cell_w - detail.width) // 2,
                              448 + (roi_h_display - detail.height) // 2))
    draw.text((gap, 422), "Center ROI enlarged with nearest-neighbor display", fill="#4b5563", font=label_font)
    draw.text((gap, 621), "Single image only. Pillow bicubic is a baseline, not the official MATLAB generator.",
              fill="#4b5563", font=small_font)
    draw.text((gap, 640), "With GT: uint8 RGB PSNR, border crop 4. This figure is not a model ranking.",
              fill="#4b5563", font=small_font)
    return canvas, roi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="RGB input image; relative paths start at the repository root")
    parser.add_argument("--gt", help="Optional matching x4 RGB ground truth")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Checkpoint path")
    parser.add_argument("--checkout", default=str(REPO / "sources/SAFMN"),
                        help="Upstream SAFMN checkout; relative paths start at the repository root")
    parser.add_argument("--weight-key", choices=("params", "params_ema"), default="params")
    parser.add_argument("--output-dir", help="Default: learning/outputs/<input-stem>_30k")
    args = parser.parse_args()

    input_path = repo_path(args.input) if args.input else DEFAULT_INPUT.resolve()
    gt_path = repo_path(args.gt) if args.gt else (DEFAULT_GT.resolve() if not args.input else None)
    weights_path = repo_path(args.weights)
    output_dir = repo_path(args.output_dir or f"learning/outputs/{input_path.stem}_30k")
    outputs = {name: (output_dir / name).resolve() for name in
               ("sr.png", "bicubic.png", "comparison.png", "result.json")}
    protected = {input_path, weights_path}
    if gt_path is not None:
        protected.add(gt_path)
    for name, path in outputs.items():
        if path in protected:
            raise ValueError(f"Output {name} would overwrite an input, GT, or weights file: {path}")

    image = read_rgb(input_path)
    if min(image.size) < 8:
        raise ValueError("SAFMN's four spatial levels require input width and height of at least 8.")
    target_size = (image.width * 4, image.height * 4)
    gt = read_rgb(gt_path) if gt_path else None
    if gt is not None and gt.size != target_size:
        raise ValueError(f"GT must be exactly x4 input size: expected {target_size}, got {gt.size}.")

    threads = min(4, os.cpu_count() or 1)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(threads)
    checkout = repo_path(args.checkout)
    model = load_safmn(checkout)(dim=36, n_blocks=8, ffn_scale=2.0, upscaling_factor=4).cpu()
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or args.weight_key not in checkpoint:
        raise ValueError(f"Checkpoint does not contain the requested weight key: {args.weight_key}")
    model.load_state_dict(checkpoint[args.weight_key], strict=True)
    if not all(torch.isfinite(value).all().item() for value in model.state_dict().values()):
        raise ValueError("Checkpoint contains non-finite values.")
    model.eval()
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).unsqueeze(0)
    label = checkpoint_label(weights_path)
    print(f"Input: {input_path}\nModel: {label} [{args.weight_key}], strict loading, CPU ({threads} threads)")
    print(f"Size: {image.width}x{image.height} -> {target_size[0]}x{target_size[1]}")
    start = time.perf_counter()
    cpu_start = time.process_time()
    with torch.inference_mode():
        prediction = model(tensor)
    inference_cpu_seconds = time.process_time() - cpu_start
    inference_wall_seconds = time.perf_counter() - start
    if not torch.isfinite(prediction).all().item():
        raise ValueError("Inference returned non-finite pixels.")
    sr_array = prediction[0].clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    sr = Image.fromarray(sr_array)
    if sr.size != target_size:
        raise ValueError(f"Unexpected model output size: {sr.size}, expected {target_size}")
    bicubic = image.resize(target_size, Image.Resampling.BICUBIC)
    panels = [("Input x4 (nearest)", image.resize(target_size, Image.Resampling.NEAREST)),
              ("Pillow bicubic", bicubic), (f"{label} [{args.weight_key}]", sr)]
    if gt is not None:
        panels.append(("GT", gt))
    comparison, roi = comparison_image(panels)
    metrics = None
    if gt is not None:
        metrics = {"protocol": "single-image uint8 RGB PSNR; crop_border=4; float64 MSE; peak=255",
                   "sr": rgb_psnr(sr, gt), "pillow_bicubic": rgb_psnr(bicubic, gt)}
    architecture_path = checkout / "basicsr/archs/safmn_arch.py"
    result = {
        "input": str(input_path), "gt": str(gt_path) if gt_path else None,
        "weights": str(weights_path), "weights_sha256": sha256(weights_path),
        "weight_key": args.weight_key,
        "model": {"name": "SAFMN", "label": label, "dim": 36, "n_blocks": 8,
                  "ffn_scale": 2.0, "scale": 4, "parameters": sum(p.numel() for p in model.parameters()),
                  "architecture_path": str(architecture_path), "architecture_sha256": sha256(architecture_path)},
        "input_size_wh": list(image.size), "output_size_wh": list(sr.size),
        "device": "cpu", "torch_threads": threads,
        "inference_wall_seconds": inference_wall_seconds, "inference_cpu_seconds": inference_cpu_seconds,
        "torch_version": str(torch.__version__), "numpy_version": np.__version__,
        "outputs": {name: str(path) for name, path in outputs.items()},
        "comparison_center_roi_xyxy": list(roi), "metrics": metrics,
        "notes": ["Single-image local practice result; not a dataset-average benchmark or model ranking.",
                  "Pillow bicubic baseline is not the official MATLAB degradation generator.",
                  "Inference timing is one CPU forward pass, without warmup; file I/O is excluded."]}
    output_dir.mkdir(parents=True, exist_ok=True)
    sr.save(outputs["sr.png"])
    bicubic.save(outputs["bicubic.png"])
    comparison.save(outputs["comparison.png"])
    outputs["result.json"].write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Forward: {inference_wall_seconds:.3f} s wall time; {inference_cpu_seconds:.3f} s CPU time")
    if metrics is not None:
        def format_psnr(value):
            return value if isinstance(value, str) else f"{value:.4f}"
        print(f"Single-image RGB PSNR, crop 4: SR {format_psnr(metrics['sr']['psnr_db'])} dB; "
              f"Pillow bicubic {format_psnr(metrics['pillow_bicubic']['psnr_db'])} dB")
    else:
        print("No GT supplied: saved images without PSNR metrics.")
    print("Pillow bicubic is a local baseline, not the official MATLAB generator.")
    print(f"Saved: {output_dir}\nView: {outputs['comparison.png']}\nRecord: {outputs['result.json']}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
