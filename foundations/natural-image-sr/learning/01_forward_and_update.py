"""CPU exercises: trace upstream SAFMN, then inspect one TinySR update.

Run from the workbench root:
    .venv-learning/bin/python learning/01_forward_and_update.py trace
    .venv-learning/bin/python learning/01_forward_and_update.py update
"""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from _safmn_source import load_safmn


class TinySR(nn.Module):
    """A small teaching network, independently written for this lesson."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3 * 4**2, kernel_size=3, padding=1),
            nn.PixelShuffle(4),
        )

    def forward(self, x):
        return self.layers(x)


def shape_hook(name):
    # A forward hook observes a module's input and output without changing them.
    def show(module, inputs, output):
        print(f"{name:28} {str(list(inputs[0].shape)):20} -> {list(output.shape)}")
    return show


def trace_safmn(args):
    SAFMN = load_safmn(args.checkout)
    model = SAFMN(dim=args.dim, n_blocks=args.blocks,
                  ffn_scale=2.0, upscaling_factor=4).eval()
    selected = (
        "to_feat",
        "feats.0.norm1",
        "feats.0.safm.mfr.0", "feats.0.safm.mfr.1",
        "feats.0.safm.mfr.2", "feats.0.safm.mfr.3",
        "feats.0.safm.aggr", "feats.0.safm",
        "feats.0.ccm.ccm.0", "feats.0.ccm.ccm.2",
        "feats.0", "feats", "to_img.0", "to_img.1",
    )
    modules = dict(model.named_modules())
    handles = [modules[name].register_forward_hook(shape_hook(name))
               for name in selected]
    lq = torch.rand(1, 3, args.height, args.width)
    print("SAFMN: random weights, CPU shape check; not an image-quality test.")
    print(f"source: {args.checkout / 'basicsr/archs/safmn_arch.py'}")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"input: {list(lq.shape)}")
    try:
        with torch.no_grad():
            sr = model(lq)
    finally:
        for handle in handles:
            handle.remove()
    expected = (1, 3, args.height * 4, args.width * 4)
    assert tuple(sr.shape) == expected, (tuple(sr.shape), expected)
    print(f"output: {list(sr.shape)}; expected shape verified")


def make_teaching_pair():
    """One synthetic pair; no benchmark images or validation score involved."""
    coord = torch.linspace(0, 1, 32)
    yy, xx = torch.meshgrid(coord, coord, indexing="ij")
    pattern = (torch.sin(6 * torch.pi * xx) * torch.cos(6 * torch.pi * yy) + 1) / 2
    gt = torch.stack((xx, yy, pattern)).unsqueeze(0)
    lq = F.interpolate(gt, size=(8, 8), mode="bicubic",
                       align_corners=False, antialias=True)
    return lq, gt


def inspect_update(args):
    model = TinySR().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lq, gt = make_teaching_pair()
    first_weight = model.layers[0].weight
    print("TinySR: synthetic training pair, CPU, L1 loss.")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"lq: {list(lq.shape)}; gt: {list(gt.shape)}")
    print("Each row reuses the same pair; it is not a validation result.")

    for step in range(1, args.steps + 1):
        # 1. Clear accumulated gradients; parameters are unchanged.
        optimizer.zero_grad(set_to_none=True)
        gradient_cleared = first_weight.grad is None
        weight_before = first_weight.detach().clone()

        # 2. Compute the prediction and its loss against the target.
        sr = model(lq)
        assert sr.shape == gt.shape
        loss = F.l1_loss(sr, gt)

        # 3. Compute gradients. backward() does not update the parameters.
        loss.backward()
        unchanged_after_backward = torch.equal(first_weight.detach(), weight_before)
        gradient_norm = first_weight.grad.norm().item()

        # 4. Let Adam update the parameters using those gradients.
        optimizer.step()
        weight_change = (first_weight.detach() - weight_before).abs().max().item()

        # 5. Inspect the new prediction without building a gradient graph.
        with torch.no_grad():
            loss_after = F.l1_loss(model(lq), gt).item()

        assert gradient_cleared and unchanged_after_backward
        assert gradient_norm > 0 and weight_change > 0
        print(f"step {step:2}: L1 {loss.item():.6f} -> {loss_after:.6f}; "
              f"first-conv grad L2={gradient_norm:.6f}; "
              f"max weight change={weight_change:.6f}")
        if step == 1:
            print(f"  gradient is None after zero_grad: {gradient_cleared}")
            print(f"  weights unchanged after backward: {unchanged_after_backward}")
            print(f"  weights changed after optimizer.step: {weight_change > 0}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("trace", "update"))
    parser.add_argument("--checkout", type=Path,
                        default=Path(__file__).resolve().parents[1] / "sources/SAFMN")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--dim", type=int, default=36)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    if args.mode == "trace" and (
        min(args.height, args.width) < 8
        or args.dim < 4 or args.dim % 4 != 0 or args.blocks < 1
    ):
        parser.error("trace needs height/width >= 8, dim divisible by 4, blocks >= 1")
    if args.steps < 1 or args.lr <= 0:
        parser.error("steps and lr must be positive")
    torch.manual_seed(10)
    torch.set_num_threads(2)
    print(f"PyTorch {torch.__version__}; CPU; seed=10")
    if args.mode == "trace":
        trace_safmn(args)
    else:
        inspect_update(args)


if __name__ == "__main__":
    main()
