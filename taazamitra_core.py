"""
TaazaMitra - shared code used by train.py, lab_reference.py and app.py

Contains: model definition, image pre-processing, L*a*b* extraction,
Grad-CAM, shelf-life / days-remaining logic.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms

IMG_SIZE = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
FRESH_CLASSES = ["fresh", "semi_fresh", "rotten"]
RIPE_CLASSES = ["unripe", "ripe", "overripe"]
STORAGE_CLASSES = ["room", "fridge"]


# --------------------------------------------------------------------------
# Image helpers
# --------------------------------------------------------------------------
def load_rgb(path_or_file) -> Image.Image:
    """Open an image, respect phone EXIF rotation, return RGB."""
    img = Image.open(path_or_file)
    if img.format == "JPEG":
        img.draft("RGB", (640, 640))          # fast decode of big phone photos
    if img.mode in ("P", "LA", "RGBA") or "transparency" in img.info:
        img = img.convert("RGBA")             # transparent cut-outs: put them on WHITE
        white = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white, img)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def eval_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def train_transform(strong: bool = False):
    """strong=True: harsher crops / perspective / blur / erasing, to cope with real-world photos."""
    t = [
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.35 if strong else 0.5, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30 if strong else 20),
    ]
    if strong:
        t += [transforms.RandomPerspective(0.25, p=0.4),
              transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 2.0))], p=0.3)]
    # hue is kept tiny on purpose: colour is the main ripeness / spoilage cue
    t += [transforms.ColorJitter(brightness=0.4 if strong else 0.3, contrast=0.4 if strong else 0.3,
                                 saturation=0.3 if strong else 0.2, hue=0.03 if strong else 0.02),
          transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    if strong:
        t.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)))
    return transforms.Compose(t)


# --------------------------------------------------------------------------
# Model: one backbone, several heads
# --------------------------------------------------------------------------
class TaazaNet(nn.Module):
    """
    Shared pretrained backbone (MobileNetV3-Large or EfficientNet-B0) with
      item_head  : which fruit / vegetable
      fresh_head : fresh / semi_fresh / rotten
      ripe_head  : unripe / ripe / overripe
      days_head  : days remaining (regression; also sees the storage condition)
    """

    def __init__(self, n_items: int, backbone: str = "mobilenet_v3_large", pretrained: bool = True):
        super().__init__()
        if backbone == "mobilenet_v3_large":
            w = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            m = models.mobilenet_v3_large(weights=w)
            feat = 960
        elif backbone == "efficientnet_b0":
            w = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            m = models.efficientnet_b0(weights=w)
            feat = 1280
        else:
            raise ValueError(f"unknown backbone {backbone}")
        self.features = m.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.2)
        self.item_head = nn.Linear(feat, n_items)
        self.fresh_head = nn.Linear(feat, len(FRESH_CLASSES))
        self.ripe_head = nn.Linear(feat, len(RIPE_CLASSES))
        self.days_head = nn.Sequential(
            nn.Linear(feat + len(STORAGE_CLASSES), 128), nn.ReLU(), nn.Linear(128, 1)
        )

    def embed(self, fmap):
        return self.drop(self.pool(fmap).flatten(1))

    def heads(self, feat, storage_onehot):
        return {
            "item": self.item_head(feat),
            "fresh": self.fresh_head(feat),
            "ripe": self.ripe_head(feat),
            "days": self.days_head(torch.cat([feat, storage_onehot], dim=1)).squeeze(1),
        }

    def forward(self, x, storage_onehot=None):
        if storage_onehot is None:
            storage_onehot = torch.zeros(x.size(0), len(STORAGE_CLASSES), device=x.device)
            storage_onehot[:, 0] = 1.0
        return self.heads(self.embed(self.features(x)), storage_onehot)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path, device=None):
    device = device or pick_device()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    model = TaazaNet(len(meta["items"]), meta["backbone"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), meta


# --------------------------------------------------------------------------
# L*a*b* colour extraction (output e)
# --------------------------------------------------------------------------
def segment_foreground(rgb: np.ndarray):
    """
    Separate the produce from the background (white cloth, table ...).
    Background colour = median of the image border; pixels far from it are produce.
    Falls back to the central region if that fails. Returns (mask_bool, method).
    """
    h, w = rgb.shape[:2]
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    b = max(2, int(0.05 * min(h, w)))
    border = np.concatenate([
        lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
        lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3),
    ])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(lab - bg, axis=2)
    d8 = np.clip(dist / max(float(dist.max()), 1e-6) * 255, 0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (d8 > thr).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == biggest).astype(np.uint8)
    frac = float(mask.mean())
    if 0.03 <= frac <= 0.90:
        return mask.astype(bool), "segmented"
    centre = np.zeros((h, w), bool)
    centre[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = True
    return centre, "centre-crop"


def lab_values(pil_img: Image.Image) -> dict:
    """Mean CIE L*a*b* of the produce pixels (L 0-100, a/b roughly -128..127)."""
    rgb = np.array(pil_img.convert("RGB").resize((256, 256)))
    mask, method = segment_foreground(rgb)
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    px = lab[mask]
    return {
        "L": float(px[:, 0].mean()), "a": float(px[:, 1].mean()), "b": float(px[:, 2].mean()),
        "L_std": float(px[:, 0].std()), "a_std": float(px[:, 1].std()), "b_std": float(px[:, 2].std()),
        "method": method, "mask": mask,
    }


# --------------------------------------------------------------------------
# Grad-CAM (explainability)
# --------------------------------------------------------------------------
def gradcam(model: TaazaNet, x: torch.Tensor, head: str, class_idx: int, storage_onehot=None):
    """Returns a (H, W) heat-map in [0, 1] for `class_idx` of `head` ('item'|'fresh'|'ripe')."""
    model.eval()
    if storage_onehot is None:
        storage_onehot = torch.zeros(1, len(STORAGE_CLASSES), device=x.device)
        storage_onehot[:, 0] = 1.0
    with torch.enable_grad():
        fmap = model.features(x)
        fmap.retain_grad()
        out = model.heads(model.embed(fmap), storage_onehot)[head]
        model.zero_grad()
        out[0, class_idx].backward()
        weights = fmap.grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * fmap).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    cam -= cam.min()
    return cam / (cam.max() + 1e-8)


def overlay_heatmap(pil_img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = np.array(pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return Image.fromarray((base * (1 - alpha) + heat * alpha).astype(np.uint8))


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def _masked_softmax(logits: torch.Tensor, allowed: list[int]) -> np.ndarray:
    m = torch.full_like(logits, -1e9)
    m[allowed] = logits[allowed]
    return torch.softmax(m, dim=0).cpu().numpy()


@torch.no_grad()
def predict(model, meta, pil_img, device, storage="room", item_override=None):
    """
    Run the model on one image. Classes with no training data for the detected
    item are removed (their output is reported as 'Data not available').
    """
    x = eval_transform()(pil_img).unsqueeze(0).to(device)
    s = torch.zeros(1, len(STORAGE_CLASSES), device=device)
    s[0, STORAGE_CLASSES.index(storage)] = 1.0
    out = model(x, s)

    items = meta["items"]
    item_p = torch.softmax(out["item"][0], 0).cpu().numpy()
    if item_override and item_override in items:
        item_idx = items.index(item_override)
        item_conf = 1.0
    else:
        item_idx = int(item_p.argmax())
        item_conf = float(item_p[item_idx])
    item = items[item_idx]
    top3 = [(items[i], float(item_p[i])) for i in np.argsort(-item_p)[:3]]

    avail = meta["availability"].get(item, {})
    f_allowed = [FRESH_CLASSES.index(c) for c in avail.get("freshness", [])]
    r_allowed = [RIPE_CLASSES.index(c) for c in avail.get("ripeness", [])]

    fresh_p = None
    if f_allowed:
        p = _masked_softmax(out["fresh"][0], f_allowed)
        fresh_p = {c: float(p[i]) for i, c in enumerate(FRESH_CLASSES)}
    ripe_p = None
    if r_allowed:
        p = _masked_softmax(out["ripe"][0], r_allowed)
        ripe_p = {c: float(p[i]) for i, c in enumerate(RIPE_CLASSES)}

    return {
        "item": item, "item_idx": item_idx, "item_conf": item_conf, "top3": top3,
        "group": meta["item_group"].get(item, "unknown"),
        "fresh_probs": fresh_p, "ripe_probs": ripe_p,
        "days_head": float(out["days"][0]) if meta.get("days_trained") else None,
        "x": x, "storage_onehot": s,
        "fresh_idx": (max(fresh_p, key=fresh_p.get) if fresh_p else None),
        "ripe_idx": (max(ripe_p, key=ripe_p.get) if ripe_p else None),
    }


# --------------------------------------------------------------------------
# Shelf life & days remaining (outputs d and f)
# --------------------------------------------------------------------------
def load_shelf_table(path="shelf_life.json") -> dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def estimate_days(item, storage, fresh_probs, ripe_probs, shelf, days_head=None):
    """
    Days remaining under normal conditions.

    If a trained regression head exists (needs day-wise data) it is used.
    Otherwise: typical shelf life (table) x expected 'life fraction' of the
    predicted freshness stage, optionally reduced when over-ripe.
    Returns dict(days, low, high, note) or None if there is no data.
    """
    entry = shelf.get("items", {}).get(item)
    if entry is None or storage not in entry or fresh_probs is None:
        return None
    lo, hi = entry[storage]
    fr = shelf.get("stage_fractions", {"fresh": 1.0, "semi_fresh": 0.3, "rotten": 0.0})
    factor = sum(p * fr.get(c, 0.0) for c, p in fresh_probs.items())
    note = "rule-based: typical shelf life x predicted freshness stage"
    if ripe_probs and "overripe" in ripe_probs:
        factor *= 1.0 - (1.0 - shelf.get("overripe_factor", 0.4)) * ripe_probs["overripe"]
    mid = (lo + hi) / 2.0
    days, low, high = mid * factor, lo * factor, hi * factor
    if days_head is not None:
        days, low, high = max(days_head, 0.0), max(days_head - 1, 0.0), days_head + 1
        note = "regression head (trained on day-wise data)"
    return {"days": float(days), "low": float(low), "high": float(high),
            "typical": (lo, hi), "note": note}


def safe_to_eat(fresh_probs, conf_floor=0.6):
    """Coarse status. An aid only - NOT a food-safety guarantee."""
    if not fresh_probs:
        return "Data not available", "grey"
    top = max(fresh_probs, key=fresh_probs.get)
    conf = fresh_probs[top]
    if conf < conf_floor:
        return "Uncertain - inspect it yourself (smell, texture, mould)", "orange"
    return {
        "fresh": ("Likely fine to eat", "green"),
        "semi_fresh": ("Eat soon - check before eating", "orange"),
        "rotten": ("Not safe - do not eat", "red"),
    }[top]