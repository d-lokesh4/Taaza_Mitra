"""
Compute the typical L*a*b* of every item x state from the labelled dataset
(used by the app for output (e): 'L, a, b values of that specific class').

  python3 lab_reference.py --root "/Volumes/LLM MODEL/Fruits_Vegetable_Dataset" --csv labels.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from taazamitra_core import lab_values, load_rgb

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--csv", default="labels.csv")
ap.add_argument("--per-class", type=int, default=80, help="images sampled per item/state")
ap.add_argument("--out", default="lab_reference.json")
a = ap.parse_args()

df = pd.read_csv(a.csv)
df["file"] = df["file"].str.strip()
ref = {}
for state_col in ("freshness", "ripeness"):
    for (item, state), g in df.dropna(subset=[state_col]).groupby(["item", state_col]):
        g = g.sample(min(len(g), a.per_class), random_state=0)
        vals = []
        for f in tqdm(g["file"], desc=f"{item}/{state}", leave=False):
            try:
                v = lab_values(load_rgb(Path(a.root) / f))
                vals.append([v["L"], v["a"], v["b"]])
            except Exception:
                continue
        if vals:
            m, s = np.mean(vals, 0), np.std(vals, 0)
            ref.setdefault(item, {})[state] = {"L": m[0], "a": m[1], "b": m[2],
                                               "L_sd": s[0], "a_sd": s[1], "b_sd": s[2], "n": len(vals)}
Path(a.out).write_text(json.dumps(ref, indent=2))
print("saved", a.out)
