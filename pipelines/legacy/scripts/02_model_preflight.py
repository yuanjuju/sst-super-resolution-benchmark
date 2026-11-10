#!/usr/bin/env python3
"""Cloud-only WMSR CUDA and single-batch verification; no real-data training."""
import argparse
import datetime
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/root/wmsr-repro"))
    args = parser.parse_args()
    root = args.root.resolve()
    result = {
        "stage": "synthetic CUDA and model preflight",
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "real_data_training": False,
        "project_root": str(root),
        "status": "running",
    }
    destination = root / "records" / "model_preflight.json"
    try:
        import torch
        import torch.nn.functional as F
        import yaml
        from focal_frequency_loss import FocalFrequencyLoss
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
        assert torch.cuda.is_available(), "CUDA is unavailable"
        torch.manual_seed(10)
        torch.cuda.manual_seed_all(10)
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = True
        result["environment"] = {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
            "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
            "packages": {name: importlib.metadata.version(name) for name in
                         ["mamba-ssm", "torchvision", "numpy", "timm", "einops", "transformers"]},
        }
        u = torch.randn(2, 8, 32, device="cuda", requires_grad=True)
        delta = torch.randn_like(u, requires_grad=True)
        A = (-torch.rand(8, 4, device="cuda")).requires_grad_()
        B = torch.randn(2, 4, 32, device="cuda", requires_grad=True)
        C = torch.randn(2, 4, 32, device="cuda", requires_grad=True)
        D = torch.randn(8, device="cuda", requires_grad=True)
        scan_args = [u, delta, A, B, C, D]
        y = selective_scan_fn(*scan_args, delta_softplus=True)
        y_ref = selective_scan_ref(*scan_args, delta_softplus=True)
        scan_error = float((y - y_ref).abs().max())
        assert torch.allclose(y, y_ref, atol=3e-4, rtol=3e-4), scan_error
        y.square().mean().backward()
        assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in scan_args)
        result["selective_scan"] = {
            "status": "pass", "shape": list(y.shape),
            "max_abs_error_vs_reference": scan_error,
            "all_input_gradients_finite": True,
        }
        print("SELECTIVE_SCAN_PASS", json.dumps(result["selective_scan"]), flush=True)
        del u, delta, A, B, C, D, y, y_ref, scan_args

        sys.path.insert(0, str(root / "upstream"))
        from models.wmsr_network import create_model
        from models.wmsr_block import dwt_init, iwt_init
        wavelet_x = torch.randn(1, 3, 24, 24, device="cuda", requires_grad=True)
        low, high = dwt_init(wavelet_x)
        wavelet_y = iwt_init(torch.cat([low, high], dim=1))
        wavelet_error = float((wavelet_y - wavelet_x).abs().max())
        assert torch.allclose(wavelet_y, wavelet_x, atol=1e-6, rtol=1e-6)
        wavelet_y.square().mean().backward()
        assert wavelet_x.grad is not None and torch.isfinite(wavelet_x.grad).all()
        result["wavelet_roundtrip"] = {"status": "pass", "max_abs_error": wavelet_error}
        del wavelet_x, wavelet_y, low, high

        configuration = yaml.safe_load((root / "upstream/configs/wmsr_x4.yml").read_text())
        model = create_model(SimpleNamespace(**configuration)).cuda().train()
        lr = torch.rand(configuration["batch_size"], 3, 24, 24, device="cuda") * 255.0
        hr = F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False).detach()
        optimizer = torch.optim.Adam(model.parameters(), lr=configuration["lr"])
        frequency_loss = FocalFrequencyLoss(loss_weight=1.0, alpha=1.0)
        before = model.head[0].weight.detach().clone()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        losses = []
        elapsed = []
        for step in range(4):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            sr = model(lr)
            assert sr.shape == hr.shape, (sr.shape, hr.shape)
            l1 = F.l1_loss(sr, hr)
            ffl = frequency_loss(sr, hr)
            loss = 0.9 * l1 + 0.1 * ffl
            assert torch.isfinite(loss), "nonfinite loss"
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            optimizer.step()
            torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - started)
            losses.append({"step": step + 1, "total": loss.item(), "l1": l1.item(), "ffl": ffl.item()})
        update = float((model.head[0].weight.detach() - before).abs().max())
        assert update > 0, "model parameters did not update"
        missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
        result["model"] = {
            "status": "pass", "upstream_modified": False,
            "wam_blocks": len(model.body),
            "parameters_total": sum(p.numel() for p in model.parameters()),
            "parameters_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "lr_shape": list(lr.shape), "hr_shape": list(hr.shape), "sr_shape": list(sr.shape),
            "precision": "fp32", "optimizer": "Adam",
            "loss": "0.9*L1 + 0.1*FocalFrequencyLoss",
            "synthetic_optimizer_steps": 4,
            "losses": losses,
            "step_seconds_excluding_first": sum(elapsed[1:]) / len(elapsed[1:]),
            "step_seconds_all": elapsed,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            "head_weight_max_abs_update": update,
            "trainable_parameters_without_gradient": missing,
        }
        result["upstream_commit"] = subprocess.check_output(
            ["git", "-C", str(root / "upstream"), "rev-parse", "HEAD"], text=True).strip()
        result["upstream_worktree_clean"] = not subprocess.check_output(
            ["git", "-C", str(root / "upstream"), "status", "--porcelain"], text=True).strip()
        result["status"] = "pass"
        result["limits"] = "Synthetic checks only; no SST data loading, validation metrics, or paper reproduction established."
        print("MODEL_PREFLIGHT_PASS", json.dumps(result["model"]), flush=True)
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        print(result["traceback"], flush=True)
    finally:
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print("REPORT", destination, result["status"], flush=True)
    return 0 if result["status"] == "pass" else 1

if __name__ == "__main__":
    raise SystemExit(main())

