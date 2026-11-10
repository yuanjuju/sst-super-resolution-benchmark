"""Build BasicSR's HR crop index after checking every HR/LR x4 pair.

Usage: python scripts/data/build_meta_info.py datasets/DF2K
"""

import argparse
from pathlib import Path
from typing import Optional

from prepare_df2k_pairs import png_dimensions


def build_index(dataset_root: Path, expected_count: Optional[int] = None) -> int:
    hr_dir = dataset_root / "HR_sub"
    lr_dir = dataset_root / "LR_bicubic" / "X4_sub"
    hr_files = sorted(hr_dir.glob("*.png"))
    lr_names = {path.name for path in lr_dir.glob("*.png")}
    hr_names = {path.name for path in hr_files}
    if not hr_files or hr_names != lr_names:
        raise ValueError(
            f"HR/LR names differ: {len(hr_names)} HR, {len(lr_names)} LR"
        )
    if expected_count is not None and len(hr_files) != expected_count:
        raise ValueError(f"Expected {expected_count} pairs, found {len(hr_files)}")

    output = dataset_root / "meta_info_HR_sub.txt"
    temporary = output.with_suffix(".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as index:
            for hr in hr_files:
                lr = lr_dir / hr.name
                if png_dimensions(hr) != (480, 480) or png_dimensions(lr) != (120, 120):
                    raise ValueError(f"Unexpected crop dimensions: {hr.name}")
                index.write(f"{hr.name} (480,480,3)\n")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return len(hr_files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--expected-count", type=int, default=None)
    args = parser.parse_args()
    count = build_index(args.dataset_root.resolve(), args.expected_count)
    print(f"Indexed {count} aligned HR/LR crop pairs", flush=True)


if __name__ == "__main__":
    main()
