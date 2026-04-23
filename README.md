# The Inspector's Eye — Bridge Crack Detection

Automated bridge-crack detection and severity reporting using YOLOv8m, YOLO11m, and YOLO26m, with an end-to-end PDF report generator. Accompanies the ICLR 2026 term paper *"The Inspector's Eye: A Comparative Study of YOLOv8, YOLO11, and YOLO26 for Automated Bridge Crack Detection and Severity Reporting."*

---

## What's in this submission

| File | Purpose |
|---|---|
| `Code_Final_training.ipynb` | The single notebook that trains, evaluates, and produces PDF reports. |
| `Results.zip` | Pre-trained checkpoints for all three models (YOLOv8m, YOLO11m, YOLO26m) plus their validation outputs. Unpack this if you want to **skip training** and go straight to evaluation/reporting. |
| `Bridge crack dataset.v10i.yolov8-obb.zip` | The Roboflow Bridge Crack dataset (v10), pre-split into train / val / test. |
| `README.md` | This file. |

---

## Important: the notebook as uploaded only trains YOLOv8m

To keep the notebook short, the training cell (cell 6) is hardcoded to `yolov8m.pt`. The evaluation cells at the bottom (cells 12, 13, 14) already point to all three trained models inside `Results.zip` and will produce the precision / recall / mAP numbers reported in the paper. If you want to reproduce *training* for YOLO11m or YOLO26m, you swap one string in cell 6 — see below.

---

## Environment

Designed and tested on **Google Colab** with a **Tesla T4 GPU** (CUDA 12). No other environment is guaranteed to work without modification — the paths below (`/content/...`) are Colab-specific.

Required packages (installed automatically by cell 0):

- `ultralytics` (8.4.40+)
- `fpdf`
- `torch` / `torchvision` (pre-installed on Colab)
- `opencv-python` (pre-installed on Colab)

---

## Option A — Evaluation only (fastest path, ~2 minutes)

Use this path if you just want to reproduce the numbers in Table 1 of the paper and/or generate the PDF inspection report, without re-training anything.

### Step 1. Open the notebook

Upload `Code_Final_training.ipynb` to Google Colab and open it. Set the runtime to **GPU (T4)** via `Runtime → Change runtime type → T4 GPU`.

### Step 2. Upload the data

In the Colab file browser (folder icon on the left), upload both zips to the `/content/` root:

- `Bridge crack dataset.v10i.yolov8-obb.zip`
- `Results.zip`

### Step 3. Run the setup cells (in order)

- **Cell 0** — installs `ultralytics` and `fpdf`
- **Cell 1** — imports and prints the Torch / CUDA version
- **Cell 2** — confirms the GPU is visible
- **Cell 3** — unzips the bridge crack dataset into `/content/dataset/`
- **Cell 4** — unzips `Results.zip` into `/content/`, producing `/content/Results/` with the three trained model folders
- **Cell 10** — points Ultralytics' dataset search path at `/content`
- **Cell 11** — clamps any out-of-bounds label coordinates (required, some test labels go slightly outside $[0,1]$)

### Step 3.5. Fix `data.yaml`

**This step is required and is easy to miss.** The `data.yaml` file that ships inside the Roboflow zip assumes a local directory layout that does not exist in Colab:

```yaml
path: ../datasets/roboflow

train: train/images
val: valid/images
test: test/images
```

Ultralytics will fail to find the images because `../datasets/roboflow` resolves to nothing on the Colab filesystem. Open `/content/dataset/data.yaml` (double-click it in the Colab file browser, or run `!cat /content/dataset/data.yaml` to inspect it) and edit the first line so the `path:` key points at your actual dataset root. Change:

```yaml
path: ../datasets/roboflow
```

to:

```yaml
path: /content/dataset
```

Leave the `train`, `val`, `test`, and `names` entries alone. The full corrected file should read:

```yaml
path: /content/dataset

train: train/images
val: valid/images
test: test/images

names:
  0: crack
  1: hole
  2: spalling
```

Save the file. If you ever re-unzip the dataset (e.g. in a new Colab session), you will have to redo this edit — the zip always ships the Roboflow-relative path.

### Step 4. SKIP the training cell

**Do not run cell 6.** That cell is the training loop and takes roughly 45 minutes on a T4. The weights it would produce are already inside `Results.zip`.

### Step 5. Run the evaluation cells

- **Cell 12** — evaluates **YOLOv8m** on the test split, prints precision / recall / mAP / F1 for the crack class
- **Cell 13** — evaluates **YOLO11m** on the test split
- **Cell 14** — evaluates **YOLO26m** on the test split

Each cell also writes a confusion matrix, PR curve, and `predictions.json` into a fresh `/content/runs/detect/val-N/` folder.

### Step 6. (Optional) Generate the PDF inspection report

- **Cell 8** — runs `analyze_infrastructure_batch` + `generate_pdf_report` and writes `Final_Inspection_Report.pdf` to a folder under `/content/runs/detect/predict*/`. By default it uses `runs/detect/train5/weights/best.pt`; edit the `MODEL_WEIGHTS` variable inside cell 8 to point at whichever checkpoint inside `Results/` you want, e.g.

  ```python
  MODEL_WEIGHTS = '/content/Results/crack_detection_results_26m/train/weights/best.pt'
  ```

- **Cell 9** — zips the full `runs/detect/` folder and downloads it to your machine.

---

## Option B — Train everything from scratch

Use this path if you want to re-train all three models yourself. Expect **roughly 45 minutes per model on a T4**, plus re-running the evaluation cells afterwards.

### ⚠ Important: directory paths will NOT match the evaluation cells

The evaluation cells 12–14 hardcode these paths (from my runs, inside `Results.zip`):

```
/content/Results/crack_detection_results_v8m/train5/weights/best.pt
/content/Results/crack_detection_results_11m/train-10/weights/best.pt
/content/Results/crack_detection_results_26m/train/weights/best.pt
```

When **you** train, Ultralytics will instead save checkpoints to `/content/runs/detect/train/`, `/content/runs/detect/train2/`, `/content/runs/detect/train3/`, etc., depending on how many times you've run training in the session. These auto-incremented directory names are not predictable, so **you must manually update the path in cells 12, 13, and 14** to point at wherever your new `best.pt` lives. The easiest way to find them after training:

```python
!find /content/runs -name 'best.pt'
```

### Step 1. Setup

Same as Option A, steps 1–3, **including the `data.yaml` edit in Step 3.5** — you must still change `path: ../datasets/roboflow` to `path: /content/dataset` or training will fail to locate the images. You do **not** need `Results.zip` for this path.

### Step 2. Train YOLOv8m

Run **cell 6** as-is. It looks like this:

```python
model = YOLO('yolov8m.pt')
results = model.train(
    data='/content/dataset/data.yaml',
    epochs=300,
    imgsz=640,
    plots=True,
    device=0,
    augment=True,
    fliplr=0.5,
    degrees=15.0,
    patience=30
)
```

When it finishes, note the output directory printed at the bottom — something like `runs/detect/train/weights/best.pt`.

### Step 3. Train YOLO11m

**Edit cell 6** and change `'yolov8m.pt'` to `'yolo11m.pt'`:

```python
model = YOLO('yolo11m.pt')   # ← changed
results = model.train(
    data='/content/dataset/data.yaml',
    epochs=300,
    imgsz=640,
    plots=True,
    device=0,
    augment=True,
    fliplr=0.5,
    degrees=15.0,
    patience=30
)
```

Run the cell again. Ultralytics will download the `yolo11m.pt` pretrained weights automatically the first time. The new `best.pt` will land in `runs/detect/train2/` (or similar).

### Step 4. Train YOLO26m

**Edit cell 6** again and change the weights string to `'yolo26m.pt'`:

```python
model = YOLO('yolo26m.pt')   # ← changed
```

Run the cell. The new `best.pt` will land in `runs/detect/train3/` (or similar).

### Step 5. Update the evaluation cells with YOUR paths

In **cell 12**, replace the `YOLO(...)` path with your YOLOv8m `best.pt` path. Example:

```python
model = YOLO('/content/runs/detect/train/weights/best.pt')   # your v8m path
```

In **cell 13**, do the same for YOLO11m:

```python
model = YOLO('/content/runs/detect/train2/weights/best.pt')  # your 11m path
```

In **cell 14**, do the same for YOLO26m:

```python
model = YOLO('/content/runs/detect/train3/weights/best.pt')  # your 26m path
```

The `train`, `train2`, `train3` suffixes are illustrative — use whatever Ultralytics actually printed at the end of each training run. Confirm with `!find /content/runs -name 'best.pt'` if unsure.

### Step 6. Run cells 11–14

- **Cell 11** — clamps labels (required).
- **Cells 12 / 13 / 14** — evaluate your three freshly trained models on the test split.

### Step 7. (Optional) Generate the PDF report

Same as Option A, Step 6. Point `MODEL_WEIGHTS` in cell 8 at whichever of your new `best.pt` files you want to use for field inference.

---

## Expected results on the test split (crack class)

These are the numbers from `Results.zip` and reported in Table 1 of the paper. Your re-trained numbers should land in the same neighborhood, with small variation due to stochastic training:

| Model | Precision | Recall | F1 | mAP@50 | mAP@50–95 |
|---|---|---|---|---|---|
| YOLOv8m | 0.3426 | 0.3077 | 0.3242 | 0.1507 | 0.0395 |
| YOLO11m | 0.3458 | 0.3846 | 0.3642 | **0.2919** | 0.0897 |
| YOLO26m | **0.4662** | **0.4035** | **0.4326** | 0.2893 | **0.0996** |

---

## Troubleshooting

**`FileNotFoundError` on `best.pt`** — you either forgot to unzip `Results.zip` (Option A) or you're pointing the evaluation cells at the old frozen paths instead of your newly-trained ones (Option B). Run `!find /content -name 'best.pt'` to see every checkpoint currently on disk.

**`dataset.yaml` not found** — cell 3 didn't run, or the zip name doesn't match. The dataset should extract to `/content/dataset/data.yaml`.

**Training / validation crashes with "images not found" or a path error** — you almost certainly skipped the `data.yaml` edit in Step 3.5. The Roboflow zip hardcodes `path: ../datasets/roboflow`, which doesn't exist in Colab. Open `/content/dataset/data.yaml` and change that line to `path: /content/dataset`.

**`cv2.imread` returns `None`** — a corrupt or non-image file slipped into the inference folder. Cell 8's glob already filters by extension, but if you added a custom folder, make sure it contains only `.jpg` / `.jpeg` / `.png` files.

**Training takes forever / OOMs** — you're probably not on a T4. Either switch runtime to T4 GPU, or lower `imgsz` from 640 to 416 in cell 6.

**Stale label cache warning** — cell 11 handles this; just re-run it.

---

## Citation

If you use this code or the accompanying report, please cite the term paper (see the `.tex` / `.pdf` in the report submission).
