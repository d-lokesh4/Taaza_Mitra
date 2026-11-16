"""
TaazaMitra - train the multi-head model (MobileNetV3-Large / EfficientNet-B0).

Usage (from the project folder):
  python3 train.py --root "/Volumes/LLM MODEL/Fruits_Vegetable_Dataset" \
                   --csv "/Volumes/LLM MODEL/Model_RP/labels.csv"

The CSV 'file' column is read relative to --root.
Optional CSV columns:  days_remaining (float)  and  storage (room|fridge)
-> if present with values, the regression head is trained too.
"""
import argparse
import json
import random
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

from taazamitra_core import (FRESH_CLASSES, RIPE_CLASSES, STORAGE_CLASSES, TaazaNet,
                             eval_transform, load_rgb, pick_device, train_transform)

IGN = -100
MIN_PER_CLASS = 20          # a class with fewer images for an item counts as "no data"


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------ data
def group_key(path: str) -> str:
    """
    Augmented copies ('aug_123_IMG_x.jpg') must stay in the same split as their
    source photo, otherwise the test set leaks. Strip the aug prefix to find it.
    """
    p = Path(path)
    stem = re.sub(r"\.(jpe?g|png|webp)$", "", p.stem, flags=re.I)   # cache files are x.png.jpg
    stem = re.sub(r"^aug_\d+_", "", stem)
    return f"{p.parent.name.strip().lower()}/{stem.lower()}"


def prepare(df, root: Path):
    df = df.copy()
    df["file"] = df["file"].str.strip()
    exists = df["file"].map(lambda f: (root / f).exists())
    if (~exists).any():
        print(f"WARNING: {(~exists).sum()} files in the CSV not found under {root} - skipped")
    df = df[exists].reset_index(drop=True)

    notes = df["notes"].fillna("")
    df.loc[notes.str.contains("freshness conflict"), "freshness"] = np.nan   # ambiguous -> ignore
    df.loc[notes.str.contains("ripeness conflict"), "ripeness"] = np.nan

    df["group_id"] = df["file"].map(group_key)
    return df


def build_meta(df, backbone, days_trained):
    items = sorted(df["item"].unique())
    item_group = df.drop_duplicates("item").set_index("item")["group"].to_dict()
    availability = {}
    for it, g in df.groupby("item"):
        fc = g["freshness"].value_counts()
        rc = g["ripeness"].value_counts()
        availability[it] = {
            "freshness": [c for c in FRESH_CLASSES if fc.get(c, 0) >= MIN_PER_CLASS],
            "ripeness": [c for c in RIPE_CLASSES if rc.get(c, 0) >= MIN_PER_CLASS],
        }
    return {"items": items, "item_group": item_group, "availability": availability,
            "backbone": backbone, "days_trained": days_trained,
            "fresh_classes": FRESH_CLASSES, "ripe_classes": RIPE_CLASSES}


class ProduceDS(Dataset):
    def __init__(self, df, root, items, tf):
        self.df, self.root, self.tf = df.reset_index(drop=True), Path(root), tf
        self.item_ix = {c: i for i, c in enumerate(items)}
        self.has_days = "days_remaining" in df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        try:
            img = load_rgb(self.root / r["file"])
        except Exception:                      # unreadable image -> use a random other one
            return self[random.randrange(len(self))]
        f = FRESH_CLASSES.index(r["freshness"]) if isinstance(r["freshness"], str) else IGN
        rp = RIPE_CLASSES.index(r["ripeness"]) if isinstance(r["ripeness"], str) else IGN
        days = float("nan")
        st = 0
        if self.has_days:
            days = float(r["days_remaining"]) if pd.notna(r["days_remaining"]) else float("nan")
            if "storage" in r and isinstance(r["storage"], str) and r["storage"] in STORAGE_CLASSES:
                st = STORAGE_CLASSES.index(r["storage"])
        return (self.tf(img), self.item_ix[r["item"]], f, rp,
                torch.tensor(days, dtype=torch.float32), st)


def split(df, seed):
    """~70/15/15 split, grouped (no leakage between augmented copies), stratified by item/state."""
    strat = (df["item"] + "|" + df["freshness"].fillna("-") + "|" + df["ripeness"].fillna("-"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        k = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
        rest_idx, test_idx = next(k.split(df, strat, df["group_id"]))
        rest = df.iloc[rest_idx]
        k2 = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed + 1)
        tr, va = next(k2.split(rest, strat.iloc[rest_idx], rest["group_id"]))
    return rest.iloc[tr], rest.iloc[va], df.iloc[test_idx]


def class_weights(series, classes):
    cnt = np.array([(series == c).sum() for c in classes], dtype=float)
    w = np.where(cnt > 0, cnt.sum() / (len(classes) * np.maximum(cnt, 1)), 0.0)
    return torch.tensor(np.clip(w, 0, 5.0), dtype=torch.float32)       # clip: avoid huge weights


# ------------------------------------------------------------------ eval
@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    P = {"item": [], "fresh": [], "ripe": []}
    T = {"item": [], "fresh": [], "ripe": []}
    dp, dt = [], []
    for x, it, f, rp, d, st in loader:
        s = torch.nn.functional.one_hot(st, len(STORAGE_CLASSES)).float().to(device)
        o = model(x.to(device), s)
        for k, y in (("item", it), ("fresh", f), ("ripe", rp)):
            P[k] += o[k].argmax(1).cpu().tolist(); T[k] += y.tolist()
        dp += o["days"].cpu().tolist(); dt += d.tolist()
    res = {}
    for k in P:
        t, p = np.array(T[k]), np.array(P[k])
        m = t != IGN
        res[k + "_acc"] = float((t[m] == p[m]).mean()) if m.any() else None
        res[k + "_f1"] = float(f1_score(t[m], p[m], average="macro")) if m.any() else None
    dp, dt = np.array(dp), np.array(dt)
    m = ~np.isnan(dt)
    res["days_mae"] = float(np.abs(dp[m] - dt[m]).mean()) if m.any() else None
    return res, P, T


def score(res):
    v = [res[k] for k in ("item_f1", "fresh_f1", "ripe_f1") if res.get(k) is not None]
    return float(np.mean(v)) if v else 0.0


def save_reports(P, T, meta, out_dir: Path, tag):
    out_dir.mkdir(parents=True, exist_ok=True)
    names = {"item": meta["items"], "fresh": FRESH_CLASSES, "ripe": RIPE_CLASSES}
    txt = []
    for k in P:
        t, p = np.array(T[k]), np.array(P[k])
        m = t != IGN
        if not m.any():
            continue
        labs = list(range(len(names[k])))
        txt.append(f"=== {k} ===\n" + classification_report(
            t[m], p[m], labels=labs, target_names=names[k], zero_division=0))
        cm = confusion_matrix(t[m], p[m], labels=labs)
        pd.DataFrame(cm, index=names[k], columns=names[k]).to_csv(out_dir / f"confusion_{tag}_{k}.csv")
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(max(4, .5 * len(labs) + 3),) * 2)
            ax.imshow(cm, cmap="Greens")
            ax.set_xticks(labs); ax.set_xticklabels(names[k], rotation=90)
            ax.set_yticks(labs); ax.set_yticklabels(names[k])
            for i in labs:
                for j in labs:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
            ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{tag}: {k}")
            fig.tight_layout(); fig.savefig(out_dir / f"confusion_{tag}_{k}.png", dpi=150); plt.close(fig)
        except ImportError:
            pass
    (out_dir / f"classification_report_{tag}.txt").write_text("\n".join(txt))
    print("\n".join(txt))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root; CSV paths are relative to it")
    ap.add_argument("--csv", default="labels.csv")
    ap.add_argument("--backbone", default="mobilenet_v3_large", choices=["mobilenet_v3_large", "efficientnet_b0"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--warmup-epochs", type=int, default=2, help="epochs with the backbone frozen")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--init", default="", help="continue from an existing taazamitra.pt (fine-tune)")
    ap.add_argument("--phone-repeat", type=int, default=1,
                    help="repeat training photos whose source != 'own' this many times (e.g. 3)")
    ap.add_argument("--strong-aug", action="store_true", help="harsher augmentation (better on real-world photos)")
    ap.add_argument("--no-pretrained", action="store_true", help="(testing only) skip ImageNet weights")
    ap.add_argument("--limit", type=int, default=0, help="(testing) use only N rows")
    a = ap.parse_args()

    seed_all(a.seed)
    device = pick_device()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print("device:", device)

    df = prepare(pd.read_csv(a.csv), Path(a.root))
    if a.limit:
        df = df.sample(a.limit, random_state=a.seed).reset_index(drop=True)
    days_trained = bool("days_remaining" in df.columns and df["days_remaining"].notna().sum() >= 30)
    meta = build_meta(df, a.backbone, days_trained)
    print(f"{len(df)} images | items: {len(meta['items'])} | days head trained: {days_trained}")

    tr, va, te = split(df, a.seed)
    if a.phone_repeat > 1 and "source" in tr.columns:
        extra = tr[tr["source"] != "own"]
        tr = pd.concat([tr] + [extra] * (a.phone_repeat - 1), ignore_index=True)
        print(f"repeated {len(extra)} new-source training photos x{a.phone_repeat}")
    pd.concat([tr.assign(split="train"), va.assign(split="val"), te.assign(split="test")]) \
        .to_csv(out / "splits.csv", index=False)
    print(f"train {len(tr)} | val {len(va)} | test {len(te)}")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    mk = lambda d, tf, sh: DataLoader(ProduceDS(d, a.root, meta["items"], tf), batch_size=a.batch,
                                      shuffle=sh, num_workers=a.workers, drop_last=sh,
                                      persistent_workers=a.workers > 0)
    tl, vl, tel = mk(tr, train_transform(a.strong_aug), True), mk(va, eval_transform(), False), mk(te, eval_transform(), False)

    model = TaazaNet(len(meta["items"]), a.backbone, pretrained=not a.no_pretrained).to(device)
    if a.init:
        old = torch.load(a.init, map_location="cpu", weights_only=False)
        same = old["meta"]["items"] == meta["items"]
        sd = {k: v for k, v in old["state_dict"].items()
              if (same or not k.startswith("item_head")) and k in model.state_dict()
              and v.shape == model.state_dict()[k].shape}
        model.load_state_dict(sd, strict=False)
        print(f"initialised from {a.init} ({len(sd)} tensors; item list {'same' if same else 'CHANGED - item head re-learned'})")
    ce_item = nn.CrossEntropyLoss(label_smoothing=0.05)
    ce_fresh = nn.CrossEntropyLoss(weight=class_weights(tr["freshness"], FRESH_CLASSES).to(device), ignore_index=IGN)
    ce_ripe = nn.CrossEntropyLoss(weight=class_weights(tr["ripeness"], RIPE_CLASSES).to(device), ignore_index=IGN)
    huber = nn.SmoothL1Loss()

    head_params = [p for n, p in model.named_parameters() if not n.startswith("features.")]
    opt = torch.optim.AdamW([{"params": model.features.parameters(), "lr": a.lr * 0.1},
                             {"params": head_params, "lr": a.lr}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(a.epochs, 1))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best, best_ep = -1, 0
    for ep in range(1, a.epochs + 1):
        frozen = ep <= a.warmup_epochs
        for p in model.features.parameters():
            p.requires_grad = not frozen
        model.train()
        if frozen:
            model.features.eval()
        t0, run = time.time(), 0.0
        for x, it, f, rp, d, st in tl:
            x, it, f, rp, d = (v.to(device) for v in (x, it, f, rp, d))
            s = torch.nn.functional.one_hot(st, len(STORAGE_CLASSES)).float().to(device)
            with torch.autocast(device.type, enabled=use_amp):
                o = model(x, s)
                loss = ce_item(o["item"], it)
                if (f != IGN).any():
                    loss = loss + ce_fresh(o["fresh"], f)
                if (rp != IGN).any():
                    loss = loss + ce_ripe(o["ripe"], rp)
                dm = ~torch.isnan(d)
                if dm.any():
                    loss = loss + 0.5 * huber(o["days"][dm].float(), d[dm])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += loss.item()
        sched.step()
        res, _, _ = run_eval(model, vl, device)
        sc = score(res)
        print(f"ep {ep:02d}/{a.epochs} {'(frozen)' if frozen else ''} loss {run / max(len(tl), 1):.3f} "
              f"| val item {res['item_acc']:.3f} fresh {res['fresh_acc']} ripe {res['ripe_acc']} "
              f"| score {sc:.3f} | {time.time() - t0:.0f}s")
        if sc > best:
            best, best_ep = sc, ep
            torch.save({"state_dict": model.state_dict(), "meta": meta}, out / "taazamitra.pt")

    # ---- final test evaluation on the best checkpoint
    ck = torch.load(out / "taazamitra.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"])
    res, P, T = run_eval(model, tel, device)
    res["best_epoch"] = best_ep
    print("\nTEST:", json.dumps(res, indent=2))
    save_reports(P, T, meta, out, "test")
    (out / "metrics_test.json").write_text(json.dumps(res, indent=2))
    print(f"\nSaved model -> {out/'taazamitra.pt'}")


if __name__ == "__main__":
    main()