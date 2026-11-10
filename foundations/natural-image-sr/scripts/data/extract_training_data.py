"""Extract the official DIV2K and the HR/x4 Flickr2K images for SAFMN.

Usage on the cloud server:
    python extract_training_data.py /root/fsas/sr-repro/datasets

The command is resumable: completed files with the expected size are skipped.
"""

from pathlib import Path
from shutil import copyfileobj, disk_usage
from sys import argv
from tarfile import open as open_tar
from zipfile import ZipFile


DIV2K_ARCHIVES = (
    "DIV2K_train_HR.zip",
    "DIV2K_train_LR_bicubic_X4.zip",
    "DIV2K_valid_HR.zip",
    "DIV2K_valid_LR_bicubic_X4.zip",
)


def copy_image(source, target: Path, expected_size: int) -> bool:
    if target.is_file() and target.stat().st_size == expected_size:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as output:
        copyfileobj(source, output, length=1024 * 1024)
    if temporary.stat().st_size != expected_size:
        raise OSError(f"Wrong extracted size: {target}")
    temporary.replace(target)
    return True


def extract_div2k(root: Path) -> None:
    for archive_name in DIV2K_ARCHIVES:
        expected_folder = archive_name.removesuffix(".zip").removesuffix("_X4")
        count = 0
        written = 0
        with ZipFile(root / "_archives" / archive_name) as archive:
            for info in archive.infolist():
                parts = Path(info.filename).parts
                if info.is_dir() or len(parts) < 2 or parts[0] != expected_folder:
                    continue
                if Path(info.filename).suffix.lower() != ".png" or ".." in parts:
                    continue
                target = root / "DIV2K" / Path(*parts)
                with archive.open(info) as source:
                    written += copy_image(source, target, info.file_size)
                count += 1
        print(f"{archive_name}: {count} images, {written} newly extracted", flush=True)


def extract_flickr(root: Path) -> None:
    archive_path = root / "_archives" / "Flickr2K.tar"
    allowed = {
        ("Flickr2K", "Flickr2K_HR"),
        ("Flickr2K", "Flickr2K_LR_bicubic", "X4"),
    }
    count = 0
    written = 0
    with open_tar(archive_path, "r:") as archive:
        for member in archive:
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            if parts[:-1] not in allowed or Path(member.name).suffix.lower() != ".png":
                continue
            if ".." in parts:
                raise ValueError(f"Unsafe archive path: {member.name}")
            source = archive.extractfile(member)
            if source is None:
                raise OSError(f"Cannot read: {member.name}")
            with source:
                written += copy_image(source, root / Path(*parts), member.size)
            count += 1
            if count % 500 == 0:
                print(f"Flickr2K: {count} images processed", flush=True)
    print(f"Flickr2K.tar: {count} images, {written} newly extracted", flush=True)


def main() -> None:
    root = Path(argv[1]).resolve()
    free_gb = disk_usage(root).free / 1e9
    if free_gb < 25:
        raise OSError(f"Only {free_gb:.1f} GB free; expected at least 25 GB")
    extract_div2k(root)
    extract_flickr(root)


if __name__ == "__main__":
    main()
