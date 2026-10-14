"""Download the MovieLens 100K dataset.

MovieLens 100K (GroupLens) — 100,000 ratings (1-5) from 943 users on 1,682
movies — is the canonical benchmark for recommender systems. We fetch the two
files we need from a public mirror into ``data/``.

Run:
    python -m src.download_data
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BASE = "https://raw.githubusercontent.com/roustask/ml-100k-NN/master/ml-100k"
FILES = {"u.data": f"{BASE}/u.data", "u.item": f"{BASE}/u.item"}


def download() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = DATA / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  exists  {name} ({dest.stat().st_size:,} bytes)")
            continue
        print(f"  fetch   {name} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"  saved   {name} ({dest.stat().st_size:,} bytes)")
    print("Done.")


if __name__ == "__main__":
    download()
