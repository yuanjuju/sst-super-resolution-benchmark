"""Download the official DF2K source archives to the project data disk.

Each HTTP range is saved separately so an interrupted download can resume.
Run on the cloud server: python download_training_data.py ARCHIVE_DIRECTORY
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import math
import os
import shutil
import sys
import time
from urllib.request import Request, urlopen


SOURCES = {
    "DIV2K_train_HR.zip": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "DIV2K_train_LR_bicubic_X4.zip": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip",
    "DIV2K_valid_HR.zip": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
    "DIV2K_valid_LR_bicubic_X4.zip": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip",
    "Flickr2K.tar": "https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar",
}
CHUNK = 64 * 1024 * 1024
WORKERS = 2  # The DIV2K origin throttled eight simultaneous connections.
RETRIES = 20


def content_length(url):
    for attempt in range(RETRIES):
        try:
            with urlopen(Request(url, method="HEAD"), timeout=45) as response:
                return int(response.headers["Content-Length"])
        except Exception:
            if attempt == RETRIES - 1:
                raise
            time.sleep(min(60, 3 * (attempt + 1)))


def fetch_part(url, part_dir, index, total):
    start = index * CHUNK
    end = min(total - 1, start + CHUNK - 1)
    size = end - start + 1
    target = part_dir / f"{index:05d}.part"
    if target.is_file() and target.stat().st_size == size:
        return
    for attempt in range(RETRIES):
        temporary = target.with_suffix(".tmp")
        try:
            request = Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urlopen(request, timeout=120) as response:
                if response.status != 206:
                    raise ValueError(f"HTTP {response.status} ignored Range request")
                expected_range = f"bytes {start}-{end}/{total}"
                if response.headers.get("Content-Range") != expected_range:
                    raise ValueError("Unexpected Content-Range")
                with temporary.open("wb") as output:
                    shutil.copyfileobj(response, output)
            if temporary.stat().st_size != size:
                raise ValueError("Incomplete range")
            os.replace(temporary, target)
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == RETRIES - 1:
                raise
            time.sleep(min(60, 3 * (attempt + 1)))


def download(name, url, directory):
    target = directory / name
    total = content_length(url)
    if target.is_file() and target.stat().st_size == total:
        print(f"EXISTS {name}: {total} bytes", flush=True)
        return target
    part_dir = directory / f".{name}.parts"
    part_dir.mkdir(exist_ok=True)
    count = math.ceil(total / CHUNK)
    print(f"START {name}: {total} bytes in {count} ranges", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(fetch_part, url, part_dir, i, total) for i in range(count)]
        done = 0
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 20 == 0 or done == count:
                print(f"PROGRESS {name}: {done}/{count} ranges", flush=True)
    temporary = directory / f".{name}.assembling"
    with temporary.open("wb") as output:
        for index in range(count):
            with (part_dir / f"{index:05d}.part").open("rb") as source:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
    if temporary.stat().st_size != total:
        raise ValueError(f"Wrong assembled size for {name}")
    os.replace(temporary, target)
    shutil.rmtree(part_dir)
    print(f"DONE {name}: {total} bytes", flush=True)
    return target


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main():
    directory = Path(sys.argv[1]).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        path = download(name, url, directory)
        digest = sha256(path)
        print(f"SHA256 {digest} {name}", flush=True)
        with (directory / "sha256.txt").open("a") as output:
            output.write(f"{digest}  {name}\n")


if __name__ == "__main__":
    main()
