"""Make SAFMN's 480/120-pixel, stride-240/60 DF2K x4 training crops.

Run from a cloud checkout containing the full official SAFMN repository:
    python scripts/data/prepare_safmn_subimages.py datasets

The project-specific .done markers make an interrupted run resumable. Source
images remain untouched. The actual crop writer is the author's script.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from shutil import disk_usage
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "SAFMN"))

import cv2  # noqa: E402

from prepare_df2k_pairs import png_dimensions  # noqa: E402
from scripts.data_preparation.extract_subimages import worker  # noqa: E402


def positions(length: int, crop: int, step: int) -> int:
    values = list(range(0, length - crop + 1, step))
    if not values:
        raise ValueError(f"Image dimension {length} is smaller than crop {crop}")
    if length - (values[-1] + crop) > 0:
        values.append(length - crop)
    return len(values)


def process_pair(args):
    hr, lr, output = args
    cv2.setNumThreads(1)
    stem = hr.stem
    marker = output / ".done" / f"{stem}.done"
    if marker.is_file():
        return int(marker.read_text()), False

    width, height = png_dimensions(hr)
    expected = positions(width, 480, 240) * positions(height, 480, 240)
    for source, folder, crop, step in (
        (hr, output / "HR_sub", 480, 240),
        (lr, output / "LR_bicubic" / "X4_sub", 120, 60),
    ):
        worker(str(source), {
            "crop_size": crop,
            "step": step,
            "thresh_size": 0,
            "save_folder": str(folder),
            "compression_level": 3,
        })

    for index in range(1, expected + 1):
        filename = f"{stem}_s{index:03d}.png"
        for folder in (output / "HR_sub", output / "LR_bicubic" / "X4_sub"):
            image = folder / filename
            if not image.is_file() or image.stat().st_size == 0:
                raise OSError(f"Missing or empty crop: {image}")
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(str(expected))
    temporary.replace(marker)
    return expected, True


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    source = root / "DF2K"
    output = source
    for folder in (output / "HR_sub", output / "LR_bicubic" / "X4_sub", output / ".done"):
        folder.mkdir(parents=True, exist_ok=True)

    free_gb = disk_usage(root).free / 1e9
    if free_gb < 65:
        raise OSError(f"Only {free_gb:.1f} GB free; need at least 65 GB to start")

    hr_images = sorted((source / "HR").glob("*.png"))
    if len(hr_images) != 3450:
        raise ValueError(f"Expected 3450 HR images, found {len(hr_images)}")
    tasks = []
    for hr in hr_images:
        lr = source / "LR_bicubic" / "X4" / f"{hr.stem}x4.png"
        if not lr.is_file():
            raise FileNotFoundError(lr)
        tasks.append((hr, lr, output))

    total_crops = 0
    newly_done = 0
    with ProcessPoolExecutor(max_workers=6) as pool:
        for start in range(0, len(tasks), 100):
            for count, is_new in pool.map(process_pair, tasks[start:start + 100]):
                total_crops += count
                newly_done += is_new
            completed = min(start + 100, len(tasks))
            free_gb = disk_usage(root).free / 1e9
            print(f"{completed}/3450 sources; {total_crops} crop pairs; {free_gb:.1f} GB free", flush=True)
            if free_gb < 8:
                raise OSError("Less than 8 GB free; stopped before the next batch")
    print(f"Done: {total_crops} pairs, {newly_done} source images newly cropped", flush=True)


if __name__ == "__main__":
    main()
