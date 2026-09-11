# Live Accident Detection

Detecting road traffic accidents in real time from live camera feeds (CCTV, dashcam, or
webcam) using object detection.

The system fine-tunes a pretrained YOLO detector on annotated accident imagery, then runs it
against a live video stream. Raw per-frame detections are passed through temporal voting so
that a sustained detection — not a single noisy frame — raises an alert, and every alert is
saved as a short video clip with pre-roll for human review.

---

## Status

**Working baseline / research prototype.** The full pipeline runs end to end and produces a
deployable model, but the bundled dataset (163 images, one class, no negatives) is a
demonstration set, not a production one. Read [Known limitations](#known-limitations) before
putting this in front of a live camera.

---

## Repository structure

```
Live-Accident-Detecction/
├── accident_detection_model.ipynb   # ← main build notebook (data → train → eval → deploy)
├── data/
│   └── raccoon_labels.csv           # master annotations (165 boxes, class "accident")
│   └── train_labels.csv             # legacy 157/8 split — superseded by the notebook
│   └── test_labels.csv
├── images/                          # the 163 source images (not tracked)
├── images_normal/                   # ← optional: accident-free frames (negatives)
├── labelImg/                        # bundled annotation GUI, for re-labelling
├── bin/protoc.exe                   # legacy TF Object Detection API tooling
├── include/google/protobuf/         # legacy TF Object Detection API tooling
├── dataset/                         # generated YOLO dataset (gitignored)
├── runs/                            # generated training outputs (gitignored)
└── alerts/                          # generated alert clips + alerts.jsonl
```

---

## Dataset

| Property | Value |
|---|---|
| Annotated boxes | 165 |
| Unique images | 163 |
| Classes | 1 (`accident`) |
| Images with >1 box | 1 |
| Formats | `.jpg`, `.png`, `.webp`, `.jpeg` |
| Image sizes | 148×162 → 2437×3688 px |
| Median box area | **71% of the frame** |
| Boxes covering >70% of frame | **87 / 165 (53%)** |
| Negative (accident-free) images | **0** |

Annotations are in Pascal-VOC-style CSV (`filename, width, height, class, xmin, ymin, xmax,
ymax`); the notebook converts them to normalised YOLO format.

**The images themselves are not tracked in git.** Place them in `images/` — filenames must
match the `filename` column, though extension and letter case may differ (the notebook
resolves both).

That "median box area" row is the most important number in the table. See
[Known limitations](#known-limitations).

---

## Quickstart

```bash
git clone https://github.com/harbidel/Live-Accident-Detecction.git
cd Live-Accident-Detecction

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U ultralytics opencv-python pandas matplotlib pyyaml jupyter

mkdir -p images                # then copy your 163 source images in here
jupyter notebook accident_detection_model.ipynb
```

Run the cells top to bottom. On a T4 GPU a full 200-epoch run takes a few minutes; on CPU,
set `EPOCHS = 30` in the config cell for a smoke test.

The notebook also runs unmodified in Google Colab — upload the repo (or mount Drive), set
`REPO_ROOT` in the config cell, and use `display=False` for inference so it writes an
annotated video instead of opening a window.

---

## What the notebook does

Everything is driven by a single config cell at the top — paths, model size, hyper-parameters,
and alert tuning all live there.

**1 · Data audit.** Loads the CSV, matches every annotation to a file on disk (tolerant of
case and extension mismatches), and reports class balance, boxes per image, and the
distribution of box area as a fraction of frame area. Tells you exactly which files are
missing if any are.

**2 · Conversion.** CSV → YOLO format with a deterministic 75/15/10 split. Two details that
matter: the split is done **per image, not per row**, so the multi-box image can't leak across
splits; and box coordinates are **rescaled against the real pixel dimensions** rather than
trusting the width/height in the CSV, which silently breaks if images were ever resized.
Optionally folds in background images from `images_normal/` (empty label file = pure
background) — the cheapest available fix for false alarms.

**3 · Training.** Fine-tunes COCO-pretrained YOLO26 with augmentation tuned for a tiny
dataset: mosaic and scale jitter to multiply effective dataset size, HSV shifts for
weather/time-of-day robustness, random erasing for occlusion, `flipud=0` (traffic cameras are
never upside down), and `close_mosaic=20` so training finishes on realistic un-stitched frames.

**4 · Evaluation.** mAP@50, mAP@50-95, precision, recall, plus the confusion matrix and
PR/F1/P/R curves rendered inline. Includes a visual pass over test predictions, because
metrics hide what failure actually looks like.

**5 · Classifier baseline (optional).** Since most boxes are near-full-frame, the notebook can
also train a binary `accident`/`normal` classifier for comparison — smaller and faster, but it
localises nothing. Requires negatives in `images_normal/`. In production the two compose well:
the cheap classifier gates the stream, the detector only runs on flagged frames.

**6 · Export.** ONNX, TensorRT, TFLite, or OpenVINO.

**7 · Live inference.** Covered below.

---

## Live inference and alerting

```python
# recorded clip → annotated mp4 (works headless / in Colab)
run_stream("test_footage.mp4", display=False, save_annotated="out.mp4")

# local webcam with preview window
run_stream(0, display=True, max_seconds=60)

# real CCTV over RTSP
run_stream("rtsp://user:password@192.168.1.64:554/Streaming/Channels/101",
           display=False, save_annotated="cctv.mp4")
```

### Why per-frame detections aren't alerts

At 25 FPS, a detector that is wrong 2% of the time produces roughly one false positive every
two seconds. `AlertSmoother` implements **N-of-M voting with a cooldown**: it fires only when
at least `ALERT_MIN_HITS` of the last `ALERT_WINDOW` frames are positive, then goes quiet for
`ALERT_COOLDOWN_S` so one incident yields one alert rather than a hundred.

| Parameter | Default | Effect |
|---|---|---|
| `CONF_THRES` | `0.35` | Per-frame detection threshold. Read the peak off `F1_curve.png`, then set it slightly **lower** — favour recall and let temporal voting handle the noise. |
| `ALERT_WINDOW` | `15` | Frames in the voting window (~0.6 s at 25 FPS). |
| `ALERT_MIN_HITS` | `8` | Positives required to fire. Raise to cut false alarms, lower to catch brief events. |
| `ALERT_COOLDOWN_S` | `30.0` | Silence after an alert. |

Each alert writes an mp4 to `alerts/` containing **pre-roll** — the seconds *before* the
trigger, held in a ring buffer — because the moment of impact always precedes the moment the
model becomes confident. Alert metadata is appended to `alerts/alerts.jsonl`. Swap
`default_alert_handler` for a webhook, MQTT publish, SMS, or database insert.

### Edge deployment

| Target | Export format | Notes |
|---|---|---|
| Any CPU, C++/.NET | `onnx` | Most portable, runs under ONNX Runtime |
| NVIDIA Jetson / dGPU | `engine` | TensorRT, fastest — must be built **on the target device** |
| Raspberry Pi, mobile | `tflite` | Use `int8=True` with calibration data |
| Intel CPU / NUC | `openvino` | Substantial CPU speedup |

YOLO26 runs NMS-free end to end, so the exported graph needs no post-processing plumbing on
the deployment side. Swap `BASE_MODEL` to `yolo26n.pt` for constrained hardware, or
`yolo11s.pt` for the older, longer-supported line.

For multi-camera setups: one process per camera, and a **bounded** frame queue so that a slow
model drops frames rather than drifting further and further behind real time.

---

## Results

Fill in after your first run — `metrics.box.map50` etc. are printed by section 4.

| Model | Images | mAP@50 | mAP@50-95 | Precision | Recall | FPS (device) |
|---|---|---|---|---|---|---|
| `yolo26s` | 163 | — | — | — | — | — |

With a test split of ~16 images, one image flipping from hit to miss moves mAP by several
points. Report these as a smoke test, not a benchmark.

---

## Re-annotating

The bundled `labelImg` GUI is set up for this:

```bash
cd labelImg
pip install -r requirements/requirements-linux-python3.txt   # or the macOS/Windows variant
make qt5py3
python labelImg.py ../images ../labelImg/data/predefined_classes.txt
```

Save in YOLO format to write `.txt` files directly, or in Pascal VOC format to produce `.xml`
that you convert to CSV. Draw boxes **tightly around the collided vehicles**, not around the
whole scene.

---

## Roadmap

Ordered by expected impact:

1. **Scale and re-source the data** — 1,000+ frames from real fixed cameras at deployment
   height and angle, covering night, rain, and glare.
2. **Add negatives** — hours of accident-free footage from the same cameras as background
   images.
3. **Re-annotate tightly** around collided vehicles only.
4. **Add temporal modelling** — track vehicles with ByteTrack (`model.track(...)`) and flag
   anomalous deceleration and trajectory intersection, or move to a video classifier
   (SlowFast, X3D, VideoMAE) over short clips. This is the largest single accuracy gain
   available and the standard approach in the accident-detection literature.
5. **Go multi-class** — `car`, `truck`, `motorcycle`, `pedestrian`, `accident` gives the model
   context and makes output far more actionable.
6. **Productionise** — TensorRT on Jetson at the edge, per-camera processes, webhook/MQTT
   integration with the existing control room stack, and a labelled feedback loop from operator
   dispositions.

> **Operational caveat.** Treat this as an operator-assist tool that surfaces clips for human
> review, never as an automatic emergency dispatcher. A missed crash and a false alarm carry
> very different costs, and only a human should close that loop.

---

## Acknowledgements

- [Ultralytics YOLO][ultra] — detection framework and pretrained weights (AGPL-3.0; commercial
  use requires an Ultralytics licence)
- [labelImg][labelimg] — bundled annotation tool (MIT)
- Project scaffolding originally derived from the [raccoon detector][raccoon] TensorFlow Object
  Detection tutorial, which is where the `raccoon_labels.csv` filename and the `protoc` tooling
  come from

[ultra]: https://github.com/ultralytics/ultralytics
[labelimg]: https://github.com/HumanSignal/labelImg
[raccoon]: https://github.com/datitran/raccoon_dataset
