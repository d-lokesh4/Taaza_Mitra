"""
One-off: shrink every image to 256x256 JPEG on your INTERNAL disk, so training
does not decode huge phone photos from the external drive every epoch.

  python3 make_cache.py --root "/Volumes/LLM MODEL/Fruits_Vegetable_Dataset" \
                        --csv  "/Volumes/LLM MODEL/Fruits_Vegetable_Dataset/labels.csv" \
                        --out  ~/taaza_cache

Then train with:  --root ~/taaza_cache --csv ~/taaza_cache/labels_cache.csv
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from taazamitra_core import load_rgb

SIZE = 256


def work(args):
    src, dst = args
    try:
        if dst.exists():
            return True
        dst.parent.mkdir(parents=True, exist_ok=True)
        img = load_rgb(src).resize((SIZE, SIZE))
        img.save(dst, "JPEG", quality=92)
        return True
    except Exception as e:
        print("skip", src, e)
        return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()

    root, out = Path(a.root), Path(a.out).expanduser()
    df = pd.read_csv(a.csv)
    df["file"] = df["file"].str.strip()
    df["new"] = df["file"] + ".jpg"            # keeps names unique (x.jpg and x.png)
    jobs = [(root / f, out / n) for f, n in zip(df["file"], df["new"])]
    with Pool(a.procs) as p:
        ok = list(tqdm(p.imap(work, jobs, chunksize=64), total=len(jobs)))
    df = df[ok].copy()
    df["file"] = df["new"]
    df.drop(columns="new").to_csv(out / "labels_cache.csv", index=False)
    print(f"done: {len(df)} images -> {out}")
