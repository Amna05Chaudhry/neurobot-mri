"""Download the LGG Segmentation dataset (Buda et al., "kaggle_3m") from a public,
login-free Hugging Face mirror and unpack it into data/raw/.

The download is resumable: if the connection drops, run the script again (it also
retries automatically) and it continues where it stopped.

Usage:  python scripts/download_data.py
"""
from __future__ import annotations

import sys
import time
import urllib.request
import zipfile
from pathlib import Path

URL = ("https://huggingface.co/datasets/gymprathap/Brain-MRI-LGG-Segmentation/"
       "resolve/main/Brain-MRI-LGG-Segmentation.zip")
EXPECTED_SIZE = 748_584_920
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ZIP = ROOT / "data" / "lgg.zip"
PART = ZIP.with_suffix(".part")
CHUNK = 1 << 20


def _fetch(max_attempts: int = 50) -> None:
    for attempt in range(1, max_attempts + 1):
        have = PART.stat().st_size if PART.exists() else 0
        if have >= EXPECTED_SIZE:
            return
        req = urllib.request.Request(URL, headers={"Range": f"bytes={have}-",
                                                   "User-Agent": "neurobot-downloader"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(PART, "ab") as f:
                if have and r.status != 206:  # server ignored Range: restart
                    f.truncate(0)
                    have = 0
                while True:
                    buf = r.read(CHUNK)
                    if not buf:
                        break
                    f.write(buf)
                    have += len(buf)
                    sys.stdout.write(f"\r  {have / 1e6:7.1f} / {EXPECTED_SIZE / 1e6:.1f} MB")
                    sys.stdout.flush()
        except Exception as e:  # noqa: BLE001 - network errors of all kinds
            print(f"\n  connection dropped ({type(e).__name__}); retry {attempt} in 3 s")
            time.sleep(3)
    if PART.stat().st_size < EXPECTED_SIZE:
        sys.exit("Download incomplete; run the script again to resume.")


def main() -> None:
    if any(RAW.rglob("*_mask.tif")):
        print(f"Dataset already present in {RAW}")
        return
    ZIP.parent.mkdir(parents=True, exist_ok=True)
    if not ZIP.exists():
        print(f"Downloading {URL}")
        _fetch()
        print()
        PART.rename(ZIP)
    print("Extracting...")
    RAW.mkdir(parents=True, exist_ok=True)
    root = RAW.resolve()
    with zipfile.ZipFile(ZIP) as zf:
        for m in zf.infolist():
            if not (RAW / m.filename).resolve().is_relative_to(root):
                raise RuntimeError(f"unsafe path in zip: {m.filename}")
        zf.extractall(RAW)
    n = len(list(RAW.rglob("*_mask.tif")))
    print(f"Done: {n} image/mask pairs in {RAW}")


if __name__ == "__main__":
    main()
