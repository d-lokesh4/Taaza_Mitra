"""
Why did the model get this photo wrong?  Runs ONE photo through the model several ways
(original, produce on white / black / grey background, tight crop, brighter, darker) and
prints how the item / freshness prediction changes. Also saves a picture of the variants.

  python3 diagnose.py strawberry.png
  python3 diagnose.py strawberry.png --ckpt checkpoints/taazamitra.pt

If 'on white' or 'on black' flips the answer, the model is relying on the BACKGROUND.
If 'brighter'/'darker' flips it, it is relying on LIGHTING.
"""
import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

from taazamitra_core import (load_checkpoint, load_rgb, pick_device, predict, segment_foreground)

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("--ckpt", default="checkpoints/taazamitra.pt")
ap.add_argument("--storage", default="room")
a = ap.parse_args()

dev = pick_device()
model, meta = load_checkpoint(a.ckpt, dev)
img = load_rgb(a.image).resize((256, 256))
arr = np.array(img)
mask, method = segment_foreground(arr)
print(f"foreground detection: {method}, produce covers {mask.mean():.0%} of the photo\n")

variants = {"original": img}
for name, col in [("on white", (255, 255, 255)), ("on black", (0, 0, 0)), ("on grey", (128, 128, 128))]:
    out = arr.copy(); out[~mask] = col
    variants[name] = Image.fromarray(out)
ys, xs = np.where(mask)
variants["tight crop"] = img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
variants["brighter"] = ImageEnhance.Brightness(img).enhance(1.4)
variants["darker"] = ImageEnhance.Brightness(img).enhance(0.7)

base_item = predict(model, meta, img, dev, a.storage)["item"]
print(f"{'variant':<12} {'item (conf)':<22} fresh   semi    rotten")
tiles = []
for name, v in variants.items():
    r = predict(model, meta, v, dev, a.storage, item_override=base_item)
    r2 = predict(model, meta, v, dev, a.storage)             # item as the model sees it
    fp = r["fresh_probs"] or {}
    item_txt = f"{r2['item']} ({r2['item_conf']:.0%})"
    print(f"{name:<12} {item_txt:<22} {fp.get('fresh', 0):<7.0%} {fp.get('semi_fresh', 0):<7.0%} {fp.get('rotten', 0):.0%}")
    t = v.resize((200, 200)).copy()
    d = ImageDraw.Draw(t)
    d.rectangle((0, 0, 200, 16), fill=(0, 0, 0))
    d.text((3, 2), f"{name}: {max(fp, key=fp.get) if fp else '-'}", fill=(255, 255, 255))
    tiles.append(t)
sheet = Image.new("RGB", (200 * len(tiles), 200))
for i, t in enumerate(tiles):
    sheet.paste(t, (200 * i, 0))
sheet.save("diagnose_output.png")
print("\nsaved diagnose_output.png (open it to see the variants)")
