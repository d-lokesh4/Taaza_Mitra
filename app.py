"""
TaazaMitra - Streamlit prototype.
Upload a photo -> fruit/vegetable, fresh/semi-fresh/rotten, ripeness, shelf life,
L*a*b* colour, days remaining, Grad-CAM heat-map.

Run:  streamlit run app.py
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from taazamitra_core import (FRESH_CLASSES, RIPE_CLASSES, estimate_days, gradcam, lab_values,
                             load_checkpoint, load_rgb, load_shelf_table, overlay_heatmap,
                             pick_device, predict, safe_to_eat)

st.set_page_config(page_title="TaazaMitra", page_icon="🌱", layout="wide")
NA = "Data not available"
HERE = Path(__file__).parent
COLORS = {"green": "#2e7d32", "orange": "#ef6c00", "red": "#c62828", "grey": "#757575"}


@st.cache_resource(show_spinner="Loading model...")
def get_model(path):
    return load_checkpoint(path, pick_device())


@st.cache_data
def get_json(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def pretty(s):
    return s.replace("_", " ").title()


def card(title, value, sub="", color=None):
    c = f"color:{COLORS[color]};" if color else ""
    st.markdown(
        f"<div style='border:1px solid #ddd;border-radius:10px;padding:10px 14px;margin-bottom:10px'>"
        f"<div style='font-size:.8rem;color:#666'>{title}</div>"
        f"<div style='font-size:1.3rem;font-weight:700;{c}'>{value}</div>"
        f"<div style='font-size:.8rem;color:#666'>{sub}</div></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🌱 TaazaMitra")
    st.caption("AI vision companion for freshness, ripeness & shelf life")
    ckpt = st.text_input("Model checkpoint", str(HERE / "checkpoints" / "taazamitra.pt"))
    storage = st.radio("Storage condition", ["room", "fridge"],
                       format_func=lambda s: {"room": "Room temperature", "fridge": "Refrigerator"}[s])
    cam_target = st.radio("Heat-map explains", ["Freshness", "Ripeness", "Item"])
    st.info("The 'safe to eat' result is an aid, **not a food-safety guarantee**.")

shelf = load_shelf_table(HERE / "shelf_life.json")
lab_ref = get_json(HERE / "lab_reference.json")

model, meta = (None, None)
if Path(ckpt).exists():
    model, meta = get_model(ckpt)
else:
    st.warning(f"No trained model at `{ckpt}` yet - run `train.py` first. "
               "Only the L*a*b* colour analysis works until then.")

# ---------------------------------------------------------------- input
tab_up, tab_cam = st.tabs(["📁 Upload photo", "📷 Take photo"])
with tab_up:
    f1 = st.file_uploader("Fruit or vegetable photo", type=["jpg", "jpeg", "png", "webp"])
with tab_cam:
    f2 = st.camera_input("Camera")
file = f2 or f1
if file is None:
    st.stop()

img = load_rgb(file)
left, right = st.columns([1, 1.4])
with left:
    st.image(img, caption="Your photo", use_container_width=True)

lab = lab_values(img)

if model is None:
    with right:
        card("(e) L*a*b* colour of the produce", f"L {lab['L']:.1f} | a {lab['a']:.1f} | b {lab['b']:.1f}",
             f"foreground: {lab['method']}")
    st.stop()

device = pick_device()
CONF_MIN = 0.75          # below this the user must confirm the item
override = None
with left:
    if st.toggle("Fix the item manually (if the model got it wrong)"):
        override = st.selectbox("Item", meta["items"], format_func=pretty)
r = predict(model, meta, img, device, storage, override)

if override is None and r["item_conf"] < CONF_MIN:
    with right:
        st.warning(f"Not sure which item this is (best guess: {pretty(r['item'])}, "
                   f"{r['item_conf']:.0%}). Please confirm it, so the stage, shelf life and "
                   "days remaining use the right item.")
        st.dataframe(pd.DataFrame(r["top3"], columns=["item", "probability"]).assign(
            item=lambda d: d["item"].map(pretty)), hide_index=True)
        picked = st.selectbox("Which item is it?", meta["items"], index=None,
                              placeholder="Choose the item...", format_func=pretty, key="confirm_item")
    if picked is None:
        st.stop()
    override = picked
    r = predict(model, meta, img, device, storage, override)

# ---------------------------------------------------------------- results
with right:
    c0, c1 = st.columns(2)
    with c0:
        card("Detected", f"{pretty(r['item'])}", f"{r['group'].title()} · confidence {r['item_conf']:.0%}")
    with c1:
        status, col = safe_to_eat(r["fresh_probs"])
        card("Safe to consume (aid only)", status, color=col)

    a, b = st.columns(2)
    with a:   # (a) fresh or rotten  + (c) semi fresh
        fp = r["fresh_probs"]
        if fp:
            fr_rot = {k: v for k, v in fp.items() if k != "semi_fresh"}
            top = max(fr_rot, key=fr_rot.get)
            card("(a) Fresh or rotten", pretty(top), f"fresh {fp['fresh']:.0%} · rotten {fp['rotten']:.0%}")
            if "semi_fresh" in meta["availability"][r["item"]]["freshness"]:
                card("(c) Semi-fresh", f"{fp['semi_fresh']:.0%}",
                     "likely semi-fresh" if max(fp, key=fp.get) == "semi_fresh" else "not the main stage")
            else:
                card("(c) Semi-fresh", NA, "no semi-fresh images for this item")
        else:
            card("(a) Fresh or rotten", NA); card("(c) Semi-fresh", NA)
    with b:   # (b) ripened / unripened
        rp = r["ripe_probs"]
        if rp:
            top = max(rp, key=rp.get)
            card("(b) Ripeness", pretty(top), " · ".join(f"{pretty(k)} {v:.0%}" for k, v in rp.items() if v > 0.005))
        else:
            card("(b) Ripeness", NA, "no ripeness data for this item")
        # (e) L*a*b*
        ref = lab_ref.get(r["item"], {}).get(r["fresh_idx"] or "", None)
        sub = f"foreground: {lab['method']}"
        if ref:
            sub += f" · typical {r['fresh_idx']}: L {ref['L']:.0f}, a {ref['a']:.0f}, b {ref['b']:.0f}"
        card("(e) L*a*b* colour", f"L {lab['L']:.1f} | a {lab['a']:.1f} | b {lab['b']:.1f}", sub)

    # (d) shelf life, (f) time remaining
    entry = shelf.get("items", {}).get(r["item"])
    d1, d2 = st.columns(2)
    with d1:
        if entry:
            lo, hi = entry[storage]
            card("(d) Shelf life (whole, fresh)", f"{lo}–{hi} days",
                 {"room": "room temperature", "fridge": "refrigerator"}[storage])
        else:
            card("(d) Shelf life", NA)
    with d2:
        est = estimate_days(r["item"], storage, r["fresh_probs"], r["ripe_probs"], shelf, r["days_head"])
        if est is None:
            card("(f) Time remaining", NA)
        elif est["days"] < 0.5:
            card("(f) Time remaining", "≈ 0 days", "past its useful life", "red")
        else:
            card("(f) Time remaining", f"≈ {est['days']:.1f} days",
                 f"range {est['low']:.1f}–{est['high']:.1f} · {est['note']}")

# ---------------------------------------------------------------- explainability
st.subheader("What did the model look at? (Grad-CAM)")
head = {"Freshness": "fresh", "Ripeness": "ripe", "Item": "item"}[cam_target]
if head == "fresh" and r["fresh_probs"]:
    idx, label = FRESH_CLASSES.index(r["fresh_idx"]), r["fresh_idx"]
elif head == "ripe" and r["ripe_probs"]:
    idx, label = RIPE_CLASSES.index(r["ripe_idx"]), r["ripe_idx"]
else:
    head, idx, label = "item", r["item_idx"], r["item"]
cam = gradcam(model, r["x"], head, idx, r["storage_onehot"])
g1, g2 = st.columns(2)
g1.image(overlay_heatmap(img, cam), caption=f"Heat-map for '{pretty(label)}' (red = most influential)")
with g2:
    st.write("**Top-3 item guesses**")
    st.dataframe(pd.DataFrame(r["top3"], columns=["item", "probability"]).assign(
        item=lambda d: d["item"].map(pretty)), hide_index=True)
    if r["fresh_probs"]:
        st.write("**Freshness probabilities**")
        st.bar_chart(pd.Series(r["fresh_probs"]))
st.caption("Time remaining is an estimate under normal conditions; real shelf life varies with temperature, "
           "humidity and handling. Not a substitute for inspecting food before eating.")