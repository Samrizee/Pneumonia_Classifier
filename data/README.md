# Data

The dataset is **not** stored in this repository (it is ~1.2 GB). Download it once:

```bash
pip install kaggle
# Kaggle -> Account -> "Create New API Token" downloads kaggle.json
mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

bash scripts/download_data.sh
```

You should end up with:

```
data/chest_xray_data/chest_xray/
├── train/{NORMAL,PNEUMONIA}/
├── test/{NORMAL,PNEUMONIA}/
└── val/{NORMAL,PNEUMONIA}/     # only 16 images - ignored by this project
```

Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

Keep the data somewhere else? Set `CHEST_XRAY_DATA=/path/to/chest_xray` or pass `--data-root`.

No Kaggle account handy? `python run_pipeline.py --synthetic` generates a small fake dataset
with the same folder layout and the same quirks (imbalance, mixed sizes, RGB + grayscale).
