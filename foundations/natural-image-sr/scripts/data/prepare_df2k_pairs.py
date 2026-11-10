"""Check official DIV2K/Flickr2K x4 pairs and link them into one DF2K folder.

Run on the cloud server after extracting the official archives:
    python prepare_df2k_pairs.py /root/fsas/sr-repro/datasets
"""

from pathlib import Path
import struct
import sys


def png_dimensions(path):
    with path.open("rb") as image:
        header = image.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Invalid PNG header: {path}")
    return struct.unpack(">II", header[16:24])


def collect_pairs(hr_dir, lr_dir, expected):
    images = sorted(hr_dir.glob("*.png"))
    if len(images) != expected:
        raise ValueError(f"Expected {expected} HR images in {hr_dir}; found {len(images)}")
    pairs = []
    for hr in images:
        lr = lr_dir / f"{hr.stem}x4.png"
        if not lr.is_file():
            raise FileNotFoundError(lr)
        hr_width, hr_height = png_dimensions(hr)
        lr_width, lr_height = png_dimensions(lr)
        if (hr_width, hr_height) != (4 * lr_width, 4 * lr_height):
            raise ValueError(f"Scale mismatch: {hr.name}, {lr.name}")
        pairs.append((hr, lr))
    return pairs


def link_pair(hr, lr, root):
    for source, target in (
        (hr, root / "HR" / hr.name),
        (lr, root / "LR_bicubic" / "X4" / lr.name),
    ):
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise ValueError(f"Wrong existing symlink: {target}")
        elif target.exists():
            raise FileExistsError(target)
        else:
            target.symlink_to(source.resolve())


def main():
    root = Path(sys.argv[1]).resolve()
    div = root / "DIV2K"
    flickr = root / "Flickr2K"
    flickr_lr_root = next(
        (path for path in (
            flickr / "Flickr2K_train_LR_bicubic",
            flickr / "Flickr2K_LR_bicubic",
        ) if path.is_dir()),
        flickr / "Flickr2K_train_LR_bicubic",
    )
    div_pairs = collect_pairs(
        div / "DIV2K_train_HR", div / "DIV2K_train_LR_bicubic" / "X4", 800
    )
    flickr_pairs = collect_pairs(
        flickr / "Flickr2K_HR", flickr_lr_root / "X4", 2650
    )
    val_pairs = collect_pairs(
        div / "DIV2K_valid_HR", div / "DIV2K_valid_LR_bicubic" / "X4", 100
    )
    output = root / "DF2K"
    (output / "HR").mkdir(parents=True, exist_ok=True)
    (output / "LR_bicubic" / "X4").mkdir(parents=True, exist_ok=True)
    for hr, lr in div_pairs + flickr_pairs:
        link_pair(hr, lr, output)
    print(f"DF2K train pairs: {len(div_pairs) + len(flickr_pairs)}")
    print(f"DIV2K validation pairs: {len(val_pairs)}")
    print(f"Combined directory: {output}")


if __name__ == "__main__":
    main()
