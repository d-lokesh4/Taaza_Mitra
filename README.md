# 🌱 TaazaMitra – freshness, ripeness & shelf-life app

Implements the system flow from the progress-review slides: one photo → fruit/vegetable →
(a) fresh/rotten, (b) ripened/unripened, (c) semi-fresh, (d) shelf life, (e) L\*a\*b\*, (f) time remaining,
plus a Grad-CAM heat-map and a "Data not available" fallback.

## Files
| file | purpose |
|---|---|
| `taazamitra_core.py` | model (MobileNetV3-L / EfficientNet-B0 + 4 heads), L\*a\*b\* extraction, Grad-CAM, days logic |
| `train.py` | transfer-learning training, leakage-safe split, accuracy / F1 / confusion matrices |
| `lab_reference.py` | typical L\*a\*b\* per item × state from your dataset (shown next to the measured value) |
| `shelf_life.json` | typical shelf life (days) per item and storage – **edit / verify with your sources** |
| `app.py` | Streamlit app |

## Run
```bash
pip install -r requirements.txt

# 1. train (paths relative to --root are taken from labels.csv "file" column)
python3 train.py --root "/Volumes/LLM MODEL/Fruits_Vegetable_Dataset" \
                 --csv  "/Volumes/LLM MODEL/Model_RP/labels.csv" --epochs 15
#    add --backbone efficientnet_b0 to compare the two backbones

# 2. (optional) colour reference per class
python3 lab_reference.py --root "/Volumes/LLM MODEL/Fruits_Vegetable_Dataset" --csv "/Volumes/LLM MODEL/Model_RP/labels.csv"

# 3. app
streamlit run app.py
```
Training writes `checkpoints/taazamitra.pt`, `splits.csv`, `metrics_test.json`,
`classification_report_test.txt` and confusion matrices (CSV + PNG) – use these in the report.

## How the pieces work
* **Multi-task, masked loss** – each photo only has the labels its folder gives (e.g. banana "Unripened" has ripeness but no freshness). Missing labels are ignored in the loss, conflicting ones (`notes` column) too.
* **Split** – ~70/15/15, stratified, *grouped* so an `aug_N_…` copy always lands in the same split as its source photo. (Same-fruit-different-angle photos can't be detected from filenames – if you can, add a real item-ID column and change `group_key`.)
* **"Data not available"** – an item/stage with < 20 training images (e.g. semi-fresh apple, ripeness of cucumber) is removed from the prediction and shown as *Data not available*. `bell_pepper/unripe` (47 images) is the only borderline class.
* **L\*a\*b\*** – produce is separated from the background (border-colour + Otsu), mean CIE L\*a\*b\* reported.
* **Days remaining** – no day-wise data exists yet, so the app uses *typical shelf life × predicted freshness stage* (and lowers it for over-ripe). When you collect day-wise photos, add `days_remaining` (and `storage`) columns to labels.csv; `train.py` then trains the regression head automatically and the app switches to it.

## Notes
* Training from scratch on CPU is slow; on a MacBook (MPS) expect a few minutes per epoch for ~32k images at 224×224. Use `--batch 32 --workers 0` if you hit memory/multiprocessing issues on macOS.
* The "safe to consume" label is an aid, not a food-safety guarantee – the app says so.
