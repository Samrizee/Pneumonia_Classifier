# Chest X-Ray Pneumonia Classifier (CNN from scratch)

A complete solution to the *CNN Image Classifier With Preprocessing* exercise: a convolutional
neural network that separates **NORMAL** from **PNEUMONIA** chest X-rays, built to deal with the
three real-world data problems baked into the
[Kaggle Chest X-Ray dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia):

| Problem | How this project handles it |
|---|---|
| Images have inconsistent sizes and mixed grayscale / RGB | Every image is converted to grayscale, resized to 150×150 and scaled to [0, 1] |
| ~3× more PNEUMONIA than NORMAL images | Balanced class weights computed from the *original* label counts and passed to `model.fit` |
| The provided `val/` folder has only 16 images | Ignored; `train/` is split 85 / 15, **stratified by class**. `test/` is used once, at the end |

Evaluation reports per-class precision / recall / F1 and a confusion matrix, not just accuracy
(an "always PNEUMONIA" model already scores well above 50 % on this data).

> **Not a medical device.** This is a learning exercise. Never use it for diagnosis.

## Project structure

```
chest-xray-cnn/
├── README.md
├── requirements.txt            # runtime dependencies
├── requirements-dev.txt        # kaggle CLI, pytest, notebook tools
├── run_pipeline.py             # one command runs Tasks 1-6 (+ bonus)
├── notebooks/
│   └── CNN_ChestXray_Solution.ipynb   # the starter notebook with every TODO solved
├── src/
│   ├── config.py               # image size, batch size, paths, seeds
│   ├── data.py                 # Task 1-3: inspect, stratified split, preprocessing, class weights
│   ├── model.py                # Task 4: the CNN (+ MobileNetV2 for the bonus)
│   ├── train.py                # Task 5: class weights + early stopping
│   ├── evaluate.py             # Task 6: report, confusion matrix, error analysis
│   ├── visualize.py            # plots
│   ├── synthetic.py            # fake demo dataset with the same quirks
│   └── pipeline.py             # glue that runs everything and writes results/
├── scripts/
│   └── download_data.sh        # Kaggle download helper
├── data/
│   └── README.md               # where the dataset goes (not committed)
├── results/                    # figures, metrics.json, writeup.md, model (generated)
├── tests/
│   └── test_pipeline.py        # unit tests for preprocessing, split, weights, model
└── .github/workflows/
    └── ci.yml                  # runs the tests + a smoke run on every push
```

## Quick start

```bash
git clone https://github.com/YOUR-USERNAME/chest-xray-cnn.git
cd chest-xray-cnn
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
```

### Option A — try it in a couple of minutes (synthetic data, no download)

```bash
python run_pipeline.py --synthetic --ablation
```

### Option B — the real thing (Kaggle data, ~1.2 GB)

```bash
# put kaggle.json in ~/.kaggle/ first (Kaggle -> Account -> Create New API Token)
bash scripts/download_data.sh
python run_pipeline.py --ablation          # add --mobilenet for the transfer-learning bonus
```

Everything is written to `results/`: `figures/*.png`, `metrics.json`, `writeup.md` (the exercise
writeup, filled in with your actual numbers) and the trained `chest_xray_cnn.keras`.

Useful flags: `--epochs N` (ceiling; early stopping decides), `--ablation` (Bonus 1: train again
without class weights), `--mobilenet` (Bonus 2: frozen MobileNetV2 + small head, needs to download
ImageNet weights), `--data-root PATH`, `--scale F` (size of the synthetic dataset).

### Notebook

```bash
jupyter notebook notebooks/CNN_ChestXray_Solution.ipynb
```

The first code cell has `USE_SYNTHETIC = True`. **Set it to `False` for the real dataset** and run
all cells. The notebook committed here was executed on the synthetic data, so its numbers are
demonstration only.

## Design decisions (the "why")

* **Resize → grayscale → normalise → augment (train only).** Batching needs constant shapes;
  X-rays have no useful colour; augmenting validation/test data would corrupt the honest estimate
  of real-world performance.
* **Class weights from the original counts.** Augmentation creates random variants on the fly; it
  does not change how common each class really is. The weights make a NORMAL mistake cost about
  3× a PNEUMONIA mistake.
* **Stratified 85/15 split, `val/` ignored.** 16 validation images cannot support decisions like
  early stopping; one image moves the metric by more than 6 points.
* **Architecture.** 3 × (Conv2D → BatchNorm → ReLU → MaxPool) with 32 → 64 → 128 filters,
  Flatten, Dropout(0.5), Dense(128), sigmoid output. BatchNorm uses `momentum=0.9` so its running
  statistics settle quickly on a small dataset.
* **Early stopping** on `val_loss` (patience 4, best weights restored), up to 25 epochs.
* **Evaluation** on the untouched `test/` folder with per-class metrics. In a medical setting a
  false negative (missed pneumonia) is usually the worse error, so both NORMAL and PNEUMONIA
  recall are reported.

## Results

Real-dataset numbers depend on your run, so they are not baked into this README — run
`python run_pipeline.py --ablation` and read `results/writeup.md`.

For reference, the **synthetic demo** (`--synthetic`, 440 train / 120 test images) lands around
0.88–0.91 accuracy with NORMAL recall around 0.82–0.87 in my runs. On this tiny, easy dataset the
effect of class weighting is small and noisy (the gain in NORMAL recall ranged from +0.00 to
+0.11 between runs, with only 45 NORMAL test images), so it shows the pipeline works but is **not**
evidence about class weighting. Measure that on the real dataset with `--ablation`.

## Deploying on GitHub

```bash
cd chest-xray-cnn
git init
git add .
git commit -m "Initial commit: chest X-ray CNN classifier"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/chest-xray-cnn.git
git push -u origin main
```

On push, **GitHub Actions** (`.github/workflows/ci.yml`) installs the dependencies, runs the unit
tests and a short end-to-end smoke test on synthetic data, and uploads the smoke-test figures as a
downloadable artifact (Actions tab → latest run → Artifacts). The dataset is never committed
(`.gitignore` excludes it).

Training on the full dataset is best done locally, on a GPU machine, or in Colab/Kaggle notebooks.

## License

MIT
